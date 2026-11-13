// ============================================================
// SparkEV - on-device training for federated learning
//
// Local training of the car's models while it is parked or
// charging. Same procedure as the notebooks and as the server's
// reference implementation (backend/federated.py):
//
//   MLP          mini-batch Adam (state reset every round),
//                MSE, pinball (quantile) or weighted
//                cross-entropy loss, xorshift32 shuffle
//   Elastic Net  proximal gradient descent (Module 4)
//
// Weights live in one RAM block per model so they can be sent
// to the server and replaced by the new global model directly.
// ============================================================

#pragma once
#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define FL_MAX_LAYERS 3
#define FL_MAX_WIDTH 64

#define FL_LOSS_MSE 0
#define FL_LOSS_PINBALL 1
#define FL_LOSS_CE 2


// Settings of one local training run
struct FLTrainCfg
{
    int loss;
    float q;                  // pinball quantile
    const float* class_w;     // cross-entropy class weights
    float lr;
    int batch;                // 0 = full batch, no shuffling
    int epochs;
};


// Trainable part of an MLP: nl dense layers, ReLU between them,
// linear output. W[l] is [sizes[l]][sizes[l+1]] row-major.
struct FLNet
{
    int nl;
    int sizes[FL_MAX_LAYERS + 1];
    float* W[FL_MAX_LAYERS];
    float* b[FL_MAX_LAYERS];
};


// Lay out an MLP's weights inside a contiguous block (W0 b0 W1 b1 ...)
static inline void fl_net_attach(FLNet& net, float* block)
{
    float* p = block;
    for (int l = 0; l < net.nl; l++) {
        net.W[l] = p;
        p += net.sizes[l] * net.sizes[l + 1];
        net.b[l] = p;
        p += net.sizes[l + 1];
    }
}


static inline int fl_net_floats(const FLNet& net)
{
    int n = 0;
    for (int l = 0; l < net.nl; l++) n += net.sizes[l] * net.sizes[l + 1] + net.sizes[l + 1];
    return n;
}


// ------------------------------------------------------------
// Random numbers (identical to backend/federated.py)
// ------------------------------------------------------------

static inline uint32_t fl_xorshift(uint32_t& s)
{
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}


static inline void fl_permutation(int* idx, int n, uint32_t& s)
{
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(fl_xorshift(s) % (uint32_t)(i + 1));
        int t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
    }
}


// ------------------------------------------------------------
// Forward pass of one sample; acts[l] holds the output of layer l
// ------------------------------------------------------------

static inline void fl_forward(const FLNet& net, const float* x,
                              float acts[FL_MAX_LAYERS][FL_MAX_WIDTH])
{
    const float* in = x;
    for (int l = 0; l < net.nl; l++) {
        int n_in = net.sizes[l], n_out = net.sizes[l + 1];
        const float* W = net.W[l];
        for (int j = 0; j < n_out; j++) {
            float acc = net.b[l][j];
            for (int i = 0; i < n_in; i++) acc += in[i] * W[i * n_out + j];
            acts[l][j] = (l < net.nl - 1 && acc < 0.0f) ? 0.0f : acc;
        }
        in = acts[l];
    }
}


// Loss of one sample and its gradient w.r.t. the output (not yet / batch)
static inline float fl_sample_loss(const FLTrainCfg& cfg, const float* out, int n_out,
                                   const float* y_f, const int8_t* y_c, int i, float* dout)
{
    if (cfg.loss == FL_LOSS_MSE) {
        float e = out[0] - y_f[i];
        if (dout) dout[0] = 2.0f * e;
        return e * e;
    }
    if (cfg.loss == FL_LOSS_PINBALL) {
        float u = y_f[i] - out[0];
        if (dout) dout[0] = (u > 0.0f) ? -cfg.q : 1.0f - cfg.q;
        return (u > 0.0f) ? cfg.q * u : (cfg.q - 1.0f) * u;
    }
    // weighted softmax cross-entropy
    int y = y_c[i];
    float top = out[0];
    for (int k = 1; k < n_out; k++) if (out[k] > top) top = out[k];
    float p[FL_MAX_WIDTH], sum = 0.0f;
    for (int k = 0; k < n_out; k++) { p[k] = expf(out[k] - top); sum += p[k]; }
    for (int k = 0; k < n_out; k++) p[k] /= sum;
    float w = cfg.class_w[y];
    if (dout) for (int k = 0; k < n_out; k++) dout[k] = (p[k] - (k == y ? 1.0f : 0.0f)) * w;
    return -w * logf(p[y] + 1e-12f);
}


// Mean loss over a data set
static float fl_eval_mlp(const FLNet& net, const float* X, const float* y_f, const int8_t* y_c,
                         int n, const FLTrainCfg& cfg)
{
    float acts[FL_MAX_LAYERS][FL_MAX_WIDTH];
    int n_out = net.sizes[net.nl], d = net.sizes[0];
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        fl_forward(net, X + (size_t)i * d, acts);
        total += fl_sample_loss(cfg, acts[net.nl - 1], n_out, y_f, y_c, i, nullptr);
    }
    return (float)(total / n);
}


// ------------------------------------------------------------
// Local training with mini-batch Adam.
// Needs 3 x (number of weights) floats of scratch RAM (gradient and
// the two Adam moments) plus an index array; returns false if the
// RAM is not available. *scratch_bytes reports what was allocated.
// ------------------------------------------------------------

static bool fl_train_mlp(FLNet& net, const float* X, const float* y_f, const int8_t* y_c,
                         int n, const FLTrainCfg& cfg, uint32_t seed, size_t* scratch_bytes)
{
    const float adam_b1 = 0.9f, adam_b2 = 0.999f, adam_eps = 1e-8f;
    int T = fl_net_floats(net);
    size_t bytes = (size_t)3 * T * sizeof(float) + (size_t)n * sizeof(int);
    float* scratch = (float*)calloc(3 * (size_t)T, sizeof(float));
    int* order = (int*)malloc((size_t)n * sizeof(int));
    if (!scratch || !order) {
        free(scratch);
        free(order);
        return false;
    }
    if (scratch_bytes) *scratch_bytes = bytes;
    float* g = scratch;
    float* m = scratch + T;
    float* v = scratch + 2 * T;
    float* params = net.W[0];          // the block is contiguous: W0 b0 W1 b1 ...

    uint32_t s = seed ? seed : 0x9E3779B9u;
    int batch = cfg.batch > 0 ? cfg.batch : n;
    int d = net.sizes[0], n_out = net.sizes[net.nl];
    float acts[FL_MAX_LAYERS][FL_MAX_WIDTH];
    float delta[FL_MAX_WIDTH], prev[FL_MAX_WIDTH];
    long t = 0;

    for (int e = 0; e < cfg.epochs; e++) {
        if (cfg.batch > 0) fl_permutation(order, n, s);
        else for (int i = 0; i < n; i++) order[i] = i;

        for (int start = 0; start < n; start += batch) {
            int nb = min(batch, n - start);
            memset(g, 0, (size_t)T * sizeof(float));

            for (int k = 0; k < nb; k++) {
                int i = order[start + k];
                const float* x = X + (size_t)i * d;
                fl_forward(net, x, acts);
                fl_sample_loss(cfg, acts[net.nl - 1], n_out, y_f, y_c, i, delta);
                for (int j = 0; j < n_out; j++) delta[j] /= nb;

                // backward, accumulating gradients
                for (int l = net.nl - 1; l >= 0; l--) {
                    int n_in = net.sizes[l], n_o = net.sizes[l + 1];
                    const float* a_in = (l == 0) ? x : acts[l - 1];
                    float* gW = g + (net.W[l] - params);
                    float* gb = g + (net.b[l] - params);
                    for (int ii = 0; ii < n_in; ii++) {
                        float a = a_in[ii];
                        if (a != 0.0f) for (int j = 0; j < n_o; j++) gW[ii * n_o + j] += a * delta[j];
                    }
                    for (int j = 0; j < n_o; j++) gb[j] += delta[j];
                    if (l > 0) {
                        const float* W = net.W[l];
                        for (int ii = 0; ii < n_in; ii++) {
                            float acc = 0.0f;
                            if (a_in[ii] > 0.0f) for (int j = 0; j < n_o; j++) acc += W[ii * n_o + j] * delta[j];
                            prev[ii] = acc;
                        }
                        memcpy(delta, prev, (size_t)n_in * sizeof(float));
                    }
                }
            }

            // Adam step
            t++;
            float c1 = (float)(1.0 - pow(0.9, (double)t));
            float c2 = (float)(1.0 - pow(0.999, (double)t));
            for (int k = 0; k < T; k++) {
                m[k] = adam_b1 * m[k] + (1.0f - adam_b1) * g[k];
                v[k] = adam_b2 * v[k] + (1.0f - adam_b2) * g[k] * g[k];
                params[k] -= cfg.lr * (m[k] / c1) / (sqrtf(v[k] / c2) + adam_eps);
            }
        }
    }
    free(scratch);
    free(order);
    return true;
}


// ------------------------------------------------------------
// Elastic Net local training (proximal gradient), Module 4.
// w = [coef(d), intercept]. Needs d + n floats of scratch RAM.
// ------------------------------------------------------------

static bool fl_train_en(float* w, int d, const float* X, const float* y, int n,
                        float alpha, float l1_ratio, float lr, int epochs, size_t* scratch_bytes)
{
    float* grad = (float*)malloc((size_t)(d + n) * sizeof(float));
    if (!grad) return false;
    if (scratch_bytes) *scratch_bytes = (size_t)(d + n) * sizeof(float);
    float* err = grad + d;
    float thr = lr * alpha * l1_ratio;
    for (int e = 0; e < epochs; e++) {
        float err_sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float p = w[d];
            for (int j = 0; j < d; j++) p += X[(size_t)i * d + j] * w[j];
            err[i] = p - y[i];
            err_sum += err[i];
        }
        for (int j = 0; j < d; j++) {
            float acc = 0.0f;
            for (int i = 0; i < n; i++) acc += X[(size_t)i * d + j] * err[i];
            grad[j] = acc / n + alpha * (1.0f - l1_ratio) * w[j];
        }
        for (int j = 0; j < d; j++) {
            float b = w[j] - lr * grad[j];
            float a = fabsf(b) - thr;
            w[j] = a > 0.0f ? (b > 0.0f ? a : -a) : 0.0f;
        }
        w[d] -= lr * (err_sum / n);
    }
    free(grad);
    return true;
}


static float fl_eval_en(const float* w, int d, const float* X, const float* y, int n)
{
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        float p = w[d];
        for (int j = 0; j < d; j++) p += X[(size_t)i * d + j] * w[j];
        float e = p - y[i];
        total += e * e;
    }
    return (float)(total / n);
}
