import json
import os
import time

import numpy as np


# ============================================================
# ON-DEVICE FEDERATED LEARNING - SERVER SIDE
#
# The ESP32 is one real car of the fleet. While it is parked or
# charging it downloads the current global model of each module,
# trains it on its own data, uploads only the model weights and
# receives the new global model. Raw data never leaves the car.
#
# The rest of the fleet is simulated here with the other clients
# of each module's notebook. In every round a sample of them
# trains locally with exactly the same procedure as the car, and
# the server aggregates all updates (FedAvg, FedAdam for Module 2,
# FedPer keeps personal heads on the clients).
#
#   task  module / model              strategy       weights exchanged
#   m1    1 battery stress MLP        FedPer         shared base
#   m2c   2 charge need MLP           FedAvg+FedAdam full network
#   m2u   2 urgency MLP               FedPer+FedAdam shared base
#   m3    3 SOH forecasting MLP       FedAvg         layers 2-3 (layer 1 frozen)
#   m4    4 energy Elastic Net        FedAvg         coefficients + intercept
#
# local_train_mlp / local_train_en are the reference of the C code
# in esp32/firmware/SparkEV_Integrated/fl_runtime.h.
#
# Setup files: models/fl/*.npz, *.json (esp32/tools/export_fl_setup.py)
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FL_DIR = os.path.join(PROJECT_ROOT, "models", "fl")

TASK_FILES = {"m1": "m1", "m2c": "m2_charge", "m2u": "m2_urgency", "m3": "m3", "m4": "m4"}
TASK_LABELS = {
    "m1": "Module 1 battery stress (FedPer MLP)",
    "m2c": "Module 2 charge need (FedAvg MLP)",
    "m2u": "Module 2 urgency (FedPer MLP)",
    "m3": "Module 3 SOH forecast (FedAvg MLP)",
    "m4": "Module 4 energy (FedAvg Elastic Net)",
}

# Simulated cars that train in each round, besides the ESP32 (partial participation,
# as in cross-device federated learning). Module 4 has only 3 other clients.
FLEET_PER_ROUND = int(os.environ.get("SPARKEV_FL_CLIENTS_PER_ROUND", "10"))

ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8


# ------------------------------------------------------------
# Random numbers shared with the ESP32 (xorshift32)
# ------------------------------------------------------------

class XorShift32:
    def __init__(self, seed):
        self.s = (seed & 0xFFFFFFFF) or 0x9E3779B9

    def next(self):
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.s = x
        return x


def permutation(n, rng):
    """Fisher-Yates shuffle, identical to fl_permutation() on the ESP32."""
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.next() % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return np.array(idx, dtype=np.int64)


# ------------------------------------------------------------
# MLP local training (reference of fl_train_mlp)
# ------------------------------------------------------------

def mlp_forward(P, X, first=0):
    """Forward pass from layer `first`; X is the input of that layer."""
    L = len(P) // 2
    acts, h = [X], X
    for l in range(first, L):
        z = h @ P[2 * l] + P[2 * l + 1]
        h = np.maximum(z, 0) if l < L - 1 else z
        acts.append(h)
    return h, acts


def loss_and_grad(out, y, cfg):
    n = len(y)
    kind = cfg["loss"]
    if kind == "mse":
        e = out[:, 0] - y
        return float(np.mean(e * e)), (2 * e / n)[:, None].astype(np.float32)
    if kind == "pinball":
        q = np.float32(cfg["q"])
        u = y - out[:, 0]
        loss = float(np.mean(np.maximum(q * u, (q - 1) * u)))
        return loss, (np.where(u > 0, -q, 1 - q) / n)[:, None].astype(np.float32)
    # weighted softmax cross-entropy
    z = out - out.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    yi = y.astype(np.int64)
    w = np.asarray(cfg["class_weights"], np.float32)[yi]
    loss = float(np.mean(w * -np.log(p[np.arange(n), yi] + 1e-12)))
    d = p.copy()
    d[np.arange(n), yi] -= 1
    return loss, (d * (w / n)[:, None]).astype(np.float32)


def eval_loss_mlp(P, X, y, cfg, first=0):
    out, _ = mlp_forward(P, X, first)
    return loss_and_grad(out, y, cfg)[0]


def local_train_mlp(P, X, y, cfg, seed, first=0):
    """Mini-batch Adam (state reset each round), as in the notebooks.
    batch 0 = full batch without shuffling (Module 1)."""
    P = [p.astype(np.float32).copy() for p in P]
    L = len(P) // 2
    idx_t = list(range(2 * first, 2 * L))
    m = {k: np.zeros_like(P[k]) for k in idx_t}
    v = {k: np.zeros_like(P[k]) for k in idx_t}
    n, batch = len(X), cfg["batch"] or len(X)
    rng = XorShift32(seed)
    lr = np.float32(cfg["lr"])
    t = 0
    for _ in range(cfg["local_epochs"]):
        order = permutation(n, rng) if cfg["batch"] else np.arange(n)
        for s in range(0, n, batch):
            b = order[s:s + batch]
            out, acts = mlp_forward(P, X[b], first)
            _, d = loss_and_grad(out, y[b], cfg)
            grads = {}
            for l in range(L - 1, first - 1, -1):
                a_in = acts[l - first]
                grads[2 * l] = a_in.T @ d
                grads[2 * l + 1] = d.sum(0)
                if l > first:
                    d = (d @ P[2 * l].T) * (acts[l - first] > 0)
            t += 1
            c1, c2 = 1 - ADAM_B1 ** t, 1 - ADAM_B2 ** t
            for k in idx_t:
                g = grads[k].astype(np.float32)
                m[k] = ADAM_B1 * m[k] + (1 - ADAM_B1) * g
                v[k] = ADAM_B2 * v[k] + (1 - ADAM_B2) * g * g
                P[k] = (P[k] - lr * (m[k] / c1) / (np.sqrt(v[k] / c2) + ADAM_EPS)).astype(np.float32)
    return P


# ------------------------------------------------------------
# Elastic Net local training (reference of fl_train_en)
# Proximal gradient descent, as in the Module 4 notebook
# ------------------------------------------------------------

def local_train_en(coef, intercept, X, y, cfg):
    beta = coef.astype(np.float32).copy()
    bias = np.float32(intercept)
    a, l1, lr = np.float32(cfg["alpha"]), np.float32(cfg["l1_ratio"]), np.float32(cfg["lr"])
    for _ in range(cfg["local_epochs"]):
        err = X @ beta + bias - y
        g = X.T @ err / np.float32(len(y)) + a * (1 - l1) * beta
        beta = beta - lr * g
        bias = np.float32(bias - lr * err.mean())
        thr = lr * a * l1
        beta = (np.sign(beta) * np.maximum(np.abs(beta) - thr, 0)).astype(np.float32)
    return beta, float(bias)


def eval_loss_en(coef, intercept, X, y):
    e = X @ coef + intercept - y
    return float(np.mean(e * e))


# ------------------------------------------------------------
# One federated task on the server
# ------------------------------------------------------------

class FederatedTask:

    def __init__(self, task):
        name = TASK_FILES[task]
        self.task = task
        self.cfg = json.load(open(os.path.join(FL_DIR, f"{name}_fl.json")))
        z = np.load(os.path.join(FL_DIR, f"{name}_fl.npz"))
        n_g = sum(1 for k in z.files if k.startswith("G"))
        self.G = [z[f"G{i}"].astype(np.float32) for i in range(n_g)]
        self.is_en = self.cfg.get("model") == "elastic_net"
        self.first = self.cfg.get("frozen_layers", 0)
        n_pers = 2 * self.cfg.get("personal_layers", 0)
        # weights exchanged with the clients: trainable and not personal
        self.exchange = list(range(2 * self.first, len(self.G) - n_pers))
        self.personal = list(range(len(self.G) - n_pers, len(self.G)))
        self.fleet = []
        for k in range(self.cfg["n_fleet"]):
            c = dict(y=z[f"fleet_y_{k}"], n=self.cfg["fleet_sizes"][k])
            c["X"] = (z[f"fleet_h_{k}"] if f"fleet_h_{k}" in z.files else z[f"fleet_X_{k}"]).astype(np.float32)
            if self.personal:
                c["head"] = [z[f"fleet_head_{k}_{j}"].astype(np.float32) for j in range(len(self.personal))]
            self.fleet.append(c)
        self.round = 0
        self.m_s = {i: np.zeros_like(self.G[i]) for i in self.exchange}
        self.v_s = {i: np.zeros_like(self.G[i]) for i in self.exchange}
        self.history = []

    # -- payloads -------------------------------------------------
    @property
    def n_floats(self):
        return int(sum(self.G[i].size for i in self.exchange))

    def payload(self):
        return b"".join(self.G[i].astype("<f4").tobytes() for i in self.exchange)

    def unpack(self, body):
        if len(body) % 4:
            raise ValueError(f"expected {self.n_floats} float32 values ({4 * self.n_floats} bytes), "
                             f"got {len(body)} bytes")
        flat = np.frombuffer(body, dtype="<f4")
        if flat.size != self.n_floats or not np.all(np.isfinite(flat)):
            raise ValueError(f"expected {self.n_floats} finite float32 values, got {flat.size}")
        out, pos = {}, 0
        for i in self.exchange:
            out[i] = flat[pos:pos + self.G[i].size].reshape(self.G[i].shape).astype(np.float32)
            pos += self.G[i].size
        return out

    # -- simulated fleet client -----------------------------------
    def _fleet_update(self, c, seed):
        if self.is_en:
            coef, b = local_train_en(self.G[0], self.G[1][0], c["X"], c["y"], self.cfg)
            return {0: coef, 1: np.array([b], np.float32)}
        P = [p.copy() for p in self.G]
        if self.personal:
            for j, i in enumerate(self.personal):
                P[i] = c["head"][j]
        P = local_train_mlp(P, c["X"], c["y"], self.cfg, seed, first=self.first)
        if self.personal:
            c["head"] = [P[i] for i in self.personal]      # stays on the client
        return {i: P[i] for i in self.exchange}

    # -- one round --------------------------------------------------
    def run_round(self, car_body, car_n, car_id="esp32"):
        t0 = time.perf_counter()
        car = self.unpack(car_body)
        self.round += 1
        rng = np.random.default_rng(1000 + self.round)
        k = min(FLEET_PER_ROUND, len(self.fleet))
        chosen = sorted(rng.choice(len(self.fleet), k, replace=False).tolist())
        updates, weights = [car], [float(car_n)]
        t_fleet = time.perf_counter()
        for j in chosen:
            updates.append(self._fleet_update(self.fleet[j], seed=self.round * 1000 + j))
            weights.append(float(self.fleet[j]["n"]))
        t_fleet = time.perf_counter() - t_fleet
        w = np.array(weights) / sum(weights)
        for i in self.exchange:
            avg = sum(wi * u[i] for wi, u in zip(w, updates)).astype(np.float32)
            if self.cfg.get("server_opt") == "adam":
                # FedAdam (Reddi et al., 2021): averaged update = pseudo-gradient
                b1, b2 = self.cfg["server_betas"]
                delta = avg - self.G[i]
                self.m_s[i] = b1 * self.m_s[i] + (1 - b1) * delta
                self.v_s[i] = b2 * self.v_s[i] + (1 - b2) * delta ** 2
                self.G[i] = (self.G[i] + self.cfg["server_lr"] * self.m_s[i]
                             / (np.sqrt(self.v_s[i]) + self.cfg["server_tau"])).astype(np.float32)
            else:
                self.G[i] = avg
        if self.personal:
            # average head, only used to start new cars (heads never leave a client)
            wf = np.array([self.fleet[j]["n"] for j in chosen], float)
            for jj, i in enumerate(self.personal):
                self.G[i] = (sum(x * self.fleet[j]["head"][jj] for x, j in zip(wf, chosen)) / wf.sum()).astype(np.float32)
        info = dict(task=self.task, round=self.round, car=car_id, car_samples=int(car_n),
                    fleet_clients=k, participants=k + 1, samples=int(sum(weights)),
                    fleet_training_ms=round(1000 * t_fleet, 1),
                    server_ms=round(1000 * (time.perf_counter() - t0), 1),
                    payload_bytes=4 * self.n_floats, strategy=self.cfg["strategy"],
                    server_opt=self.cfg.get("server_opt", "avg"))
        self.history.append(info)
        return info


# ------------------------------------------------------------
# Server state
# ------------------------------------------------------------

TASKS = {}
for _t in TASK_FILES:
    if os.path.exists(os.path.join(FL_DIR, f"{TASK_FILES[_t]}_fl.json")):
        TASKS[_t] = FederatedTask(_t)

latest_fl_report = None

print(f"Federated learning server ready: {', '.join(TASKS)} "
      f"({FLEET_PER_ROUND} simulated cars per round besides the ESP32)")


def fl_status():
    return dict(
        tasks={t: dict(label=TASK_LABELS[t], round=s.round, strategy=s.cfg["strategy"],
                       payload_bytes=4 * s.n_floats, fleet_size=len(s.fleet),
                       last_round=s.history[-1] if s.history else None)
               for t, s in TASKS.items()},
        fleet_per_round=FLEET_PER_ROUND,
        latest_report=latest_fl_report,
    )
