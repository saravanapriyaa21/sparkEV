// ============================================================
// SparkEV - the car's federated learning tasks
//
// One task per federated model:
//
//   m1   Module 1 battery stress MLP     FedPer   base exchanged, head personal
//   m2c  Module 2 charge need MLP        FedAvg   full network exchanged
//   m2u  Module 2 urgency MLP            FedPer   base exchanged, head personal
//   m3   Module 3 SOH forecasting MLP    FedAvg   layers 2-3 exchanged (layer 1 frozen)
//   m4   Module 4 energy Elastic Net     FedAvg   coefficients + intercept exchanged
//
// Each task keeps its weights in one RAM block, laid out as
// [exchanged weights | personal weights]. After the first round the
// Module 2, 3 and 4 predictions use these updated weights.
//
// Include after model2_esp32_generated.h, model4_esp32_generated.h,
// fl_esp32_generated.h and fl_runtime.h.
// ============================================================

#pragma once

#define FL_NUM_TASKS 5
#define FL_T_M1 0
#define FL_T_M2C 1
#define FL_T_M2U 2
#define FL_T_M3 3
#define FL_T_M4 4

static const char* FL_TASK_IDS[FL_NUM_TASKS] = {"m1", "m2c", "m2u", "m3", "m4"};
static const char* FL_TASK_NAMES[FL_NUM_TASKS] = {
    "M1 stress", "M2 charge", "M2 urgency", "M3 SOH", "M4 energy"};


struct FLTask
{
    float* block;        // RAM weights, nullptr until first use
    int exchange;        // floats sent / received
    int total;           // floats in the block
    FLNet net;           // trainable layers (MLP tasks)
    FLTrainCfg cfg;
    bool is_en;          // Elastic Net (Module 4)
};

static FLTask fl_task[FL_NUM_TASKS];

// first-layer activations of the Module 3 car data (layer 1 is frozen)
static float* fl_m3_hidden = nullptr;


// ------------------------------------------------------------
// The car's local data (never leaves the car)
// ------------------------------------------------------------

struct FLCarData
{
    const float* X;
    const float* y_f;
    const int8_t* y_c;
    int n;
};

#define FL_PICK(name17, name24) (driver == 17 ? (name17) : (name24))

static FLCarData fl_car_data(int t, int driver)
{
    switch (t) {
    case FL_T_M1:
        return {FL_PICK(fl_m1_car17_x, fl_m1_car24_x), FL_PICK(fl_m1_car17_y, fl_m1_car24_y), nullptr,
                FL_PICK(FL_M1_CAR17_N, FL_M1_CAR24_N)};
    case FL_T_M2C:
        return {FL_PICK(fl_m2c_car17_x, fl_m2c_car24_x), FL_PICK(fl_m2c_car17_y, fl_m2c_car24_y), nullptr,
                FL_PICK(FL_M2C_CAR17_N, FL_M2C_CAR24_N)};
    case FL_T_M2U:
        return {FL_PICK(fl_m2u_car17_x, fl_m2u_car24_x), nullptr, FL_PICK(fl_m2u_car17_y, fl_m2u_car24_y),
                FL_PICK(FL_M2U_CAR17_N, FL_M2U_CAR24_N)};
    case FL_T_M3:
        // training input = frozen first-layer activations
        return {fl_m3_hidden, FL_PICK(fl_m3_car17_y, fl_m3_car24_y), nullptr,
                FL_PICK(FL_M3_CAR17_N, FL_M3_CAR24_N)};
    default:
        return {FL_PICK(fl_m4_car17_x, fl_m4_car24_x), FL_PICK(fl_m4_car17_y, fl_m4_car24_y), nullptr,
                FL_PICK(FL_M4_CAR17_N, FL_M4_CAR24_N)};
    }
}


// ------------------------------------------------------------
// Module 3 forecast with the (possibly updated) federated MLP
// ------------------------------------------------------------

static const float* m3_active_upper = fl_m3_factory_upper;   // W1 b1 W2 b2 (layers 2-3)

// Row by row, so the 357 x 64 weights are read from flash in order
// (reading them column by column is about 10x slower on the ESP32)
static inline void fl_m3_hidden1(const float* x, float* h)
{
    for (int j = 0; j < FL_M3_H1; j++) h[j] = fl_m3_b0[j];
    for (int i = 0; i < FL_M3_IN; i++) {
        float xi = x[i];
        if (xi == 0.0f) continue;
        const float* w = fl_m3_w0 + i * FL_M3_H1;
        for (int j = 0; j < FL_M3_H1; j++) h[j] += xi * w[j];
    }
    for (int j = 0; j < FL_M3_H1; j++) h[j] = h[j] > 0.0f ? h[j] : 0.0f;
}

// SOH in 10 cycles (fraction) from the processed input of the newest cycle
static inline float model3_soh_forecast(const float* x, float soh_now)
{
    float h1[FL_M3_H1], h2[32];
    fl_m3_hidden1(x, h1);
    const float* W1 = m3_active_upper;
    const float* b1 = W1 + FL_M3_H1 * 32;
    const float* W2 = b1 + 32;
    const float* b2 = W2 + 32;
    for (int j = 0; j < 32; j++) {
        float acc = b1[j];
        for (int i = 0; i < FL_M3_H1; i++) acc += h1[i] * W1[i * 32 + j];
        h2[j] = acc > 0.0f ? acc : 0.0f;
    }
    float out = b2[0];
    for (int i = 0; i < 32; i++) out += h2[i] * W2[i];
    return soh_now + out * FL_M3_DELTA_STD + FL_M3_DELTA_MEAN;
}


// ------------------------------------------------------------
// Set-up: weight block from the deployed (factory) model
// ------------------------------------------------------------

static void fl_set_cfg(FLTask& k, int loss, float q, const float* cw, float lr, int batch, int epochs)
{
    k.cfg.loss = loss;
    k.cfg.q = q;
    k.cfg.class_w = cw;
    k.cfg.lr = lr;
    k.cfg.batch = batch;
    k.cfg.epochs = epochs;
}

static float* fl_copy(float* dst, const float* src, int n)
{
    memcpy(dst, src, (size_t)n * sizeof(float));
    return dst + n;
}

// Allocates the task's weight block once and fills it with the deployed
// model (and the car's own FedPer head). Returns false if RAM is short.
static bool fl_init_task(int t, int driver)
{
    FLTask& k = fl_task[t];
    if (k.block) return true;
    float* p;
    switch (t) {
    case FL_T_M1:
        k.net = {3, {FL_M1_IN, 32, 16, 1}};
        k.exchange = FL_M1_EXCHANGE;
        k.total = FL_M1_TOTAL;
        fl_set_cfg(k, FL_M1_LOSS, 0.0f, nullptr, FL_M1_LR, FL_M1_BATCH, FL_M1_EPOCHS);
        if (!(k.block = (float*)malloc(k.total * sizeof(float)))) return false;
        fl_copy(k.block, fl_m1_factory, k.total);
        break;
    case FL_T_M2C:
        k.net = {3, {M2A_INPUT_DIM, MODEL2_HIDDEN1, MODEL2_HIDDEN2, 1}};
        k.total = k.exchange = fl_net_floats(k.net);
        fl_set_cfg(k, FL_M2C_LOSS, FL_M2C_Q, nullptr, FL_M2C_LR, FL_M2C_BATCH, FL_M2C_EPOCHS);
        if (!(k.block = (float*)malloc(k.total * sizeof(float)))) return false;
        p = fl_copy(k.block, m2a_w0, M2A_INPUT_DIM * MODEL2_HIDDEN1);
        p = fl_copy(p, m2a_b0, MODEL2_HIDDEN1);
        p = fl_copy(p, m2a_w1, MODEL2_HIDDEN1 * MODEL2_HIDDEN2);
        p = fl_copy(p, m2a_b1, MODEL2_HIDDEN2);
        p = fl_copy(p, m2a_w2, MODEL2_HIDDEN2);
        fl_copy(p, m2a_b2, 1);
        break;
    case FL_T_M2U:
        k.net = {3, {M2B_INPUT_DIM, MODEL2_HIDDEN1, MODEL2_HIDDEN2, MODEL2_NUM_URGENCY}};
        k.total = fl_net_floats(k.net);
        k.exchange = k.total - (MODEL2_HIDDEN2 * MODEL2_NUM_URGENCY + MODEL2_NUM_URGENCY);
        fl_set_cfg(k, FL_M2U_LOSS, 0.0f, fl_m2u_class_weights, FL_M2U_LR, FL_M2U_BATCH, FL_M2U_EPOCHS);
        if (!(k.block = (float*)malloc(k.total * sizeof(float)))) return false;
        p = fl_copy(k.block, m2b_w0, M2B_INPUT_DIM * MODEL2_HIDDEN1);
        p = fl_copy(p, m2b_b0, MODEL2_HIDDEN1);
        p = fl_copy(p, m2b_w1, MODEL2_HIDDEN1 * MODEL2_HIDDEN2);
        p = fl_copy(p, m2b_b1, MODEL2_HIDDEN2);
        fl_copy(p, FL_PICK(fl_m2u_car17_head, fl_m2u_car24_head), MODEL2_HIDDEN2 * MODEL2_NUM_URGENCY + MODEL2_NUM_URGENCY);
        break;
    case FL_T_M3:
        k.net = {2, {FL_M3_H1, 32, 1}};
        k.total = k.exchange = FL_M3_EXCHANGE;
        fl_set_cfg(k, FL_M3_LOSS, 0.0f, nullptr, FL_M3_LR, FL_M3_BATCH, FL_M3_EPOCHS);
        if (!(k.block = (float*)malloc(k.total * sizeof(float)))) return false;
        fl_copy(k.block, fl_m3_factory_upper, k.total);
        break;
    default:
        k.is_en = true;
        k.net.nl = 0;
        k.total = k.exchange = FL_M4_EXCHANGE;
        if (!(k.block = (float*)malloc(k.total * sizeof(float)))) return false;
        fl_copy(k.block, m4_energy_coef, FL_M4_FEATURES);
        k.block[FL_M4_FEATURES] = M4_ENERGY_INTERCEPT;
        k.cfg.lr = FL_M4_LR;
        k.cfg.epochs = FL_M4_EPOCHS;
        break;
    }
    if (!k.is_en) fl_net_attach(k.net, k.block);
    return true;
}


// The car's predictions use the task's RAM weights from now on
static void fl_activate(int t)
{
    FLTask& k = fl_task[t];
    if (!k.block) return;
    switch (t) {
    case FL_T_M2C:
        for (int l = 0; l < 3; l++) {
            m2a_active[2 * l] = k.net.W[l];
            m2a_active[2 * l + 1] = k.net.b[l];
        }
        break;
    case FL_T_M2U:
        for (int l = 0; l < 2; l++) {
            m2b_active[2 * l] = k.net.W[l];
            m2b_active[2 * l + 1] = k.net.b[l];
        }
        break;
    case FL_T_M3:
        m3_active_upper = k.block;
        break;
    case FL_T_M4:
        m4_energy_coef_active = k.block;
        m4_energy_intercept_active = k.block[FL_M4_FEATURES];
        break;
    default:
        break;   // Module 1 predictions stay with XGBoost
    }
}


// Urgency head used for predictions: the car's trained head once it exists
static inline const float* fl_m2u_head_w(const float* fallback)
{
    return fl_task[FL_T_M2U].block ? fl_task[FL_T_M2U].net.W[2] : fallback;
}

static inline const float* fl_m2u_head_b(const float* fallback)
{
    return fl_task[FL_T_M2U].block ? fl_task[FL_T_M2U].net.b[2] : fallback;
}


// ------------------------------------------------------------
// Inputs of a session (Module 3: frozen first layer computed once)
// ------------------------------------------------------------

static bool fl_prepare_inputs(int t, int driver, size_t* bytes)
{
    if (t != FL_T_M3) return true;
    int n = FL_PICK(FL_M3_CAR17_N, FL_M3_CAR24_N);
    const float* X = FL_PICK(fl_m3_car17_x, fl_m3_car24_x);
    fl_m3_hidden = (float*)malloc((size_t)n * FL_M3_H1 * sizeof(float));
    if (!fl_m3_hidden) return false;
    if (bytes) *bytes = (size_t)n * FL_M3_H1 * sizeof(float);
    for (int i = 0; i < n; i++) fl_m3_hidden1(X + (size_t)i * FL_M3_IN, fl_m3_hidden + (size_t)i * FL_M3_H1);
    return true;
}

static void fl_release_inputs(int t)
{
    if (t == FL_T_M3) {
        free(fl_m3_hidden);
        fl_m3_hidden = nullptr;
    }
}


// ------------------------------------------------------------
// Local loss and local training on the car's data
// ------------------------------------------------------------

static float fl_task_loss(int t, int driver)
{
    FLTask& k = fl_task[t];
    FLCarData c = fl_car_data(t, driver);
    if (k.is_en) return fl_eval_en(k.block, FL_M4_FEATURES, c.X, c.y_f, c.n);
    return fl_eval_mlp(k.net, c.X, c.y_f, c.y_c, c.n, k.cfg);
}

static bool fl_task_train(int t, int driver, uint32_t seed, size_t* scratch_bytes)
{
    FLTask& k = fl_task[t];
    FLCarData c = fl_car_data(t, driver);
    if (k.is_en)
        return fl_train_en(k.block, FL_M4_FEATURES, c.X, c.y_f, c.n, FL_M4_ALPHA, FL_M4_L1_RATIO,
                           k.cfg.lr, k.cfg.epochs, scratch_bytes);
    return fl_train_mlp(k.net, c.X, c.y_f, c.y_c, c.n, k.cfg, seed, scratch_bytes);
}

static inline int fl_task_samples(int t, int driver)
{
    return fl_car_data(t, driver).n;
}


// ------------------------------------------------------------
// Back to the deployed (factory) models: frees all weight blocks
// (used by the benchmark, which trains from the factory weights)
// ------------------------------------------------------------

static void fl_reset_all()
{
    for (int t = 0; t < FL_NUM_TASKS; t++) {
        free(fl_task[t].block);
        fl_task[t] = FLTask();
    }
    m2a_active[0] = m2a_w0; m2a_active[1] = m2a_b0;
    m2a_active[2] = m2a_w1; m2a_active[3] = m2a_b1;
    m2a_active[4] = m2a_w2; m2a_active[5] = m2a_b2;
    m2b_active[0] = m2b_w0; m2b_active[1] = m2b_b0;
    m2b_active[2] = m2b_w1; m2b_active[3] = m2b_b1;
    m3_active_upper = fl_m3_factory_upper;
    m4_energy_coef_active = m4_energy_coef;
    m4_energy_intercept_active = M4_ENERGY_INTERCEPT;
}
