# snn_phase3_step3_online_Bplus_adaptive.py
# Plan B+ con controles adaptativos y realimentación (bio-plausible)

import numpy as np
from collections import deque, Counter
from pathlib import Path
import csv

rng = np.random.default_rng(7)

# =========================
# TTFS / PSP
# =========================
class PSPKernel:
    def __init__(self, tau=15e-3, tau_s=5e-3, dt=1e-3, T=0.15):
        self.dt = dt
        self.t = np.arange(0.0, T + 1e-12, dt, dtype=np.float32)
        tau = np.float32(tau); tau_s = np.float32(tau_s)
        k = np.exp(-self.t/tau) - np.exp(-self.t/tau_s)
        k[k < 0] = 0.0
        if k.max() > 0: k /= k.max()
        self.k = k.astype(np.float32)

def latency_code(x, t_min=5, t_max=100, jitter_std=3.5, drop_p=0.08):
    t_idx = -np.ones_like(x, dtype=np.int32)
    m = x > 0
    if m.any():
        t = t_min + (1.0 - x[m]) * (t_max - t_min)
        if jitter_std > 0:
            t += rng.normal(0.0, jitter_std, size=t.shape).astype(np.float32)
        t = np.clip(t, t_min, t_max)
        if drop_p > 0:
            keep = rng.random(t.shape) > drop_p
            t[~keep] = -1
        t_idx[m] = np.floor(t).astype(np.int32)
    return t_idx

def ttfs_strength_from_times(t_idx, t_min, t_max):
    s = np.zeros_like(t_idx, dtype=np.float32)
    m = t_idx >= 0
    if m.any():
        s[m] = 1.0 - (t_idx[m] - t_min) / float(max(1, (t_max - t_min)))
        s = np.clip(s, 0.0, 1.0)
    return s

# =========================
# L1: Conv STDP + k-WTA
# =========================
class ConvSTDPKWTA:
    def __init__(self, in_hw=(16,16), k=5, stride=2, n_filters=16, k_wta=1,
                 wmin=0.02, wmax=0.6, a_plus=0.015, a_minus=0.012,
                 t_min_in=5, t_max_in=100, post_win_scale=0.02, target_norm=1.0):
        self.H, self.W = in_hw
        self.K = k; self.S = stride; self.F = n_filters
        self.k_wta = k_wta
        self.wmin = np.float32(wmin); self.wmax = np.float32(wmax)
        self.a_plus = np.float32(a_plus); self.a_minus = np.float32(a_minus)
        self.t_min_in = t_min_in; self.t_max_in = t_max_in
        self.out_h = (self.H - self.K)//self.S + 1
        self.out_w = (self.W - self.K)//self.S + 1
        self.w = rng.uniform(0.05, 0.15, size=(self.F, self.K, self.K)).astype(np.float32)
        self._norm_all_filters(target=target_norm)
        self.post_win_scale = np.float32(post_win_scale)
        self.target_norm = np.float32(target_norm)
        self.learn_enabled = True  # se puede congelar en runtime

    def _norm_all_filters(self, target=1.0):
        for f in range(self.F): self._norm_filter(f, target)
    def _norm_filter(self, f, target=1.0):
        wf = self.w[f].reshape(-1)
        n = np.linalg.norm(wf) + 1e-8
        self.w[f] = (wf * (target / n)).reshape(self.K, self.K)

    def _conv_acts(self, Smap):
        act = np.zeros((self.F, self.out_h, self.out_w), dtype=np.float32)
        for oy in range(self.out_h):
            for ox in range(self.out_w):
                y0 = oy*self.S; x0 = ox*self.S
                patch = Smap[y0:y0+self.K, x0:x0+self.K]
                act[:, oy, ox] = np.tensordot(self.w, patch, axes=((1,2),(0,1)))
        return act

    def forward_and_learn(self, t_idx_2d, boost=False):
        Smap = ttfs_strength_from_times(t_idx_2d, self.t_min_in, self.t_max_in).astype(np.float32)
        act = self._conv_acts(Smap)
        if not self.learn_enabled:
            return act
        a_plus = self.a_plus * (1.5 if boost else 1.0)
        a_minus = self.a_minus * (1.5 if boost else 1.0)
        for oy in range(self.out_h):
            for ox in range(self.out_w):
                f_win = int(np.argmax(act[:, oy, ox]))
                patch = Smap[oy*self.S:oy*self.S+self.K, ox*self.S:ox*self.S+self.K]
                w = self.w[f_win]
                dw = a_plus*(self.wmax - w)*patch - a_minus*(w - self.wmin)*patch
                w_new = np.clip(w + dw, float(self.wmin), float(self.wmax))
                nrm = np.linalg.norm(w_new.reshape(-1)) + 1e-8
                scale = 1.0 + self.post_win_scale*(self.target_norm/nrm - 1.0)
                self.w[f_win] = np.clip(w_new * scale, float(self.wmin), float(self.wmax))
        return act

    def forward_no_learn(self, t_idx_2d):
        Smap = ttfs_strength_from_times(t_idx_2d, self.t_min_in, self.t_max_in).astype(np.float32)
        return self._conv_acts(Smap)

    def activations_to_ttfs(self, act_maps, t_min_out=5, t_max_out=90):
        A = act_maps
        a_min = float(A.min()); a_max = float(A.max())
        if a_max <= a_min + 1e-8: A_norm = np.zeros_like(A, dtype=np.float32)
        else: A_norm = (A - a_min) / (a_max - a_min + 1e-8)
        t_idx = t_min_out + (1.0 - A_norm) * (t_max_out - t_min_out)
        t_idx = np.floor(t_idx).astype(np.int32)
        return t_idx.reshape(-1), int(t_max_out + 50)

# =========================
# Inhibición lateral
# =========================
def lateral_inhibition(act_maps, ksize=3, gamma=0.15):
    F, H, W = act_maps.shape
    out = act_maps.copy()
    for f in range(F):
        A = act_maps[f]
        Ap = np.pad(A, 1, mode='edge')
        neigh = (
            Ap[0:H,0:W] + Ap[0:H,1:W+1] + Ap[0:H,2:W+2] +
            Ap[1:H+1,0:W] + Ap[1:H+1,1:W+1] + Ap[1:H+1,2:W+2] +
            Ap[2:H+2,0:W] + Ap[2:H+2,1:W+1] + Ap[2:H+2,2:W+2]
        ) / 9.0
        out[f] = np.maximum(0.0, A - gamma*neigh)
    return out

# =========================
# Util
# =========================
def l2_normalize_rows(W, target=1.0):
    nrm = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    return W * np.minimum(1.0, target / nrm)

# =========================
# L2: R-STDP (actor) + homeostasis + SFA + vía supervisada débil
# =========================
class RSTDPDecisionPop:
    def __init__(self, n_inputs, n_classes, m_per_class, kernel: PSPKernel,
                 v_th=1.0, eta=1.5e-3, a_plus=1.0, a_minus=1.0,
                 tau_pre=20e-3, tau_post=20e-3, lam=0.99, leak=0.98,
                 wmin=0.02, wmax=0.8, dt=1e-3,
                 homeo_target_rate=None, homeo_lr=4e-3, syn_scale_lr=6e-5,
                 sfa_tau=800.0, sfa_delta=0.08):
        self.C = n_classes; self.M = m_per_class; self.N = n_inputs
        self.K = kernel
        self.v_th_base = np.float32(v_th)
        self.v_th = np.full(self.C, float(v_th), dtype=np.float32)
        self.sfa = np.zeros(self.C*self.M, dtype=np.float32)
        self.sfa_tau = np.float32(sfa_tau); self.sfa_delta = np.float32(sfa_delta)
        self.eta = np.float32(eta); self.eta0 = np.float32(eta)
        self.a_plus = np.float32(a_plus); self.a_minus = np.float32(a_minus)
        self.alpha_pre = np.exp(-dt/tau_pre).astype(np.float32)
        self.alpha_post = np.exp(-dt/tau_post).astype(np.float32)
        self.lam = np.float32(lam); self.leak = np.float32(leak)
        self.wmin = np.float32(wmin); self.wmax = np.float32(wmax)
        self.n_out = self.C * self.M
        self.W = l2_normalize_rows(rng.normal(0.0, 0.12, size=(self.n_out, self.N)).astype(np.float32), 1.0)
        self.baseline = np.zeros(self.C, dtype=np.float32); self.bl_beta = 0.02
        if homeo_target_rate is None: homeo_target_rate = 1.0/float(self.C)
        self.homeo_target = np.float32(homeo_target_rate)
        self.homeo_lr = np.float32(homeo_lr); self.syn_scale_lr = np.float32(syn_scale_lr)
        self.win = 200; self.win_counts = np.zeros(self.C, dtype=np.int32); self.win_total = 0

        # buffers/logs
        self.reset_episode_states()
        self.last_x_pre = np.zeros(self.N, dtype=np.float32)
        self.n_nudges = 0
        self.log_e_norm_pred = 0.0
        self.log_e_norm_nonpred = 0.0
        self.last_margin = 0.0
        self.last_top2 = (0.0, 0.0)  # (Vwin, V2)

    def reset_episode_states(self):
        self.e = np.zeros((self.n_out, self.N), dtype=np.float32)
        self.x_pre = np.zeros(self.N, dtype=np.float32)
        self.x_post = np.zeros(self.n_out, dtype=np.float32)
        self.has_spiked = np.zeros(self.n_out, dtype=np.bool_)
        self.first_spike_time = np.full(self.n_out, 1<<30, dtype=np.int32)
        self.Vmem = np.zeros(self.n_out, dtype=np.float32)

    def _build_Vin(self, times_per_input, T_steps):
        K = self.K.k; Lk = len(K)
        Vin = np.zeros((self.N, T_steps), dtype=np.float32)
        for i in range(self.N):
            s = int(times_per_input[i])
            if s < 0: continue
            n = min(Lk, T_steps - s)
            if n > 0: Vin[i, s:s+n] += K[:n]
        return Vin

    def simulate_episode(self, times_per_input, T_steps, y_true=None, teacher_nudge=False):
        """Devuelve (pred_class, t_win, T_steps, used_nudge). Guarda last_margin (top-2)."""
        self.reset_episode_states()
        self.sfa *= (1.0 - 1.0 / self.sfa_tau)
        Vin = self._build_Vin(times_per_input, T_steps)

        winner_found = False; pred_class = 0; t_win = T_steps
        top2_margin = 0.0; top_val = 0.0; second_val = 0.0

        for t in range(T_steps):
            self.x_pre *= self.alpha_pre; self.x_post *= self.alpha_post
            self.e *= self.lam; self.Vmem *= self.leak
            pre_mask = (times_per_input == t)
            if np.any(pre_mask): self.x_pre[pre_mask] += 1.0
            self.Vmem += self.W @ Vin[:, t]

            post_mask = np.zeros(self.n_out, dtype=np.bool_)
            if not winner_found:
                thr_class = np.repeat(self.v_th, self.M)
                thr_eff = thr_class + self.sfa
                over = (self.Vmem >= thr_eff) & (~self.has_spiked)
                if np.any(over):
                    cands = np.flatnonzero(over)
                    j_star = cands[int(np.argmax(self.Vmem[cands]))]
                    post_mask[j_star] = True
                    self.has_spiked[j_star] = True
                    self.first_spike_time[j_star] = t
                    winner_found = True; pred_class = int(j_star // self.M)
                    t_win = t; self.sfa[j_star] += float(self.sfa_delta)
                    # margen top-2 en el instante de decisión
                    vals = self.Vmem.copy()
                    vals[~over] = self.Vmem[~over]  # no importa, ya calculamos
                    top_val = float(vals[j_star])
                    vals[j_star] = -1e9
                    second_val = float(np.max(vals))
                    denom = abs(top_val) + abs(second_val) + 1e-6
                    top2_margin = max(0.0, (top_val - second_val) / denom)

            if np.any(post_mask): self.x_post[post_mask] += 1.0
            if np.any(post_mask):
                self.e += self.a_plus * (post_mask.astype(np.float32)[:, None] * self.x_pre[None, :])
            if np.any(pre_mask):
                self.e -= self.a_minus * (self.x_post[:, None] * pre_mask.astype(np.float32)[None, :])

        if not winner_found:
            j = int(np.argmax(self.Vmem)); pred_class = j // self.M; t_win = T_steps
            top_val = float(self.Vmem[j]); tmp = self.Vmem.copy(); tmp[j] = -1e9
            second_val = float(np.max(tmp))
            denom = abs(top_val) + abs(second_val) + 1e-6
            top2_margin = max(0.0, (top_val - second_val) / denom)
            self.sfa[j] += float(self.sfa_delta) * 0.5

        self.last_x_pre = self.x_pre.copy()
        self.last_margin = float(top2_margin)
        self.last_top2 = (top_val, second_val)

        used_nudge = False
        if (y_true is not None) and teacher_nudge:
            used_nudge = True
            self.n_nudges += 1
        return pred_class, t_win, T_steps, used_nudge

    def apply_reward_actor_only(self, R, pred_class):
        e_masked = np.zeros_like(self.e)
        start = pred_class * self.M; end = (pred_class + 1) * self.M
        e_masked[start:end, :] = self.e[start:end, :]
        self.log_e_norm_pred = float(np.linalg.norm(e_masked[start:end, :]))
        nonpred = np.vstack([self.e[:start, :], self.e[end:, :]]) if (start > 0 or end < self.n_out) else np.zeros((0, self.N), dtype=np.float32)
        self.log_e_norm_nonpred = float(np.linalg.norm(nonpred))

        R_adv = R - self.baseline[pred_class]
        self.W += self.eta * R_adv * e_masked
        self.W = np.clip(self.W, float(self.wmin), float(self.wmax))
        self.W = l2_normalize_rows(self.W, 1.0)
        self.baseline[pred_class] = (1.0 - self.bl_beta) * self.baseline[pred_class] + self.bl_beta * R
        self.W = (1.0 - self.syn_scale_lr) * self.W + self.syn_scale_lr * l2_normalize_rows(self.W, 1.0)

        # homeostasis por tasa de ganadores
        self.win_counts[pred_class] += 1; self.win_total += 1
        if self.win_total >= self.win:
            rate = self.win_counts / float(self.win_total + 1e-8)
            self.v_th += self.homeo_lr * (rate - self.homeo_target)
            self.v_th = np.clip(self.v_th, 0.7*self.v_th_base, 1.6*self.v_th_base)
            self.win_counts[:] = 0; self.win_total = 0

    def apply_supervised_nudge(self, y_true, eta_sup=5e-4, cap_norm=0.4):
        if y_true is None: return
        idx_pop = np.arange(y_true*self.M, (y_true+1)*self.M, dtype=np.int32)
        delta = eta_sup * self.last_x_pre[None, :]
        # cap por norma para estabilidad
        nrm = float(np.linalg.norm(delta))
        if nrm > cap_norm:
            delta *= (cap_norm / (nrm + 1e-8))
        self.W[idx_pop, :] += delta
        self.W = np.clip(self.W, float(self.wmin), float(self.wmax))
        self.W[idx_pop, :] = l2_normalize_rows(self.W[idx_pop, :], 1.0)

# =========================
# Feedback top-down
# =========================
class TopDownFeedback:
    def __init__(self, n_classes, F_maps, Oh, Ow, eta_f=3e-4, fmin=-0.8, fmax=0.8, beta=0.60):
        self.C = n_classes; self.F = F_maps; self.Oh = Oh; self.Ow = Ow
        self.eta_f0 = np.float32(eta_f); self.eta_f = np.float32(eta_f)
        self.fmin = np.float32(fmin); self.fmax = np.float32(fmax)
        self.beta = np.float32(beta)
        self.Fw = rng.normal(0.0, 0.05, size=(self.C, self.F, self.Oh, self.Ow)).astype(np.float32)

    def modulate(self, act_maps, cls_pred):
        return act_maps * (1.0 + self.beta * self.Fw[cls_pred])

    def update(self, act_maps_after_inhib, cls_pred, R):
        A = act_maps_after_inhib
        a_min = float(A.min()); a_max = float(A.max())
        pre = np.zeros_like(A, dtype=np.float32) if a_max<=a_min+1e-8 else (A - a_min) / (a_max - a_min + 1e-8)
        dF = self.eta_f * R * pre
        self.Fw[cls_pred] = np.clip(self.Fw[cls_pred] + dF, float(self.fmin), float(self.fmax))
        # anti-saturación ligera por filtro
        for f in range(self.F):
            m = np.max(np.abs(self.Fw[cls_pred, f])) + 1e-8
            if m > 0.8:
                self.Fw[cls_pred, f] *= (0.8 / m)

    def max_inf_norm(self):
        return float(np.max(np.abs(self.Fw)))

# =========================
# Generador continuo & evaluación
# =========================
def make_multiclass_generator(n_features=256, n_classes=10, k_on=40, sep=0.6,
                              noise=0.20, distractor_ratio=0.35, seed=11):
    rng_local = np.random.default_rng(seed)
    stride = max(1, (n_features - k_on)//n_classes)
    protos = []
    for c in range(n_classes):
        p = np.zeros(n_features, dtype=np.float32)
        start = c * stride; idx = np.arange(start, start + k_on) % n_features
        p[idx] = 1.0; protos.append(p)
    protos = np.stack(protos, axis=0)
    while True:
        y = int(rng_local.integers(0, n_classes))
        x = sep*protos[y] + (1.0 - sep)*protos[int(rng_local.integers(0, n_classes))]
        n_dist = int(distractor_ratio * k_on)
        if n_dist > 0:
            idxd = rng_local.integers(0, n_features, size=n_dist)
            x[idxd] = np.maximum(x[idxd], 0.3)
        x = np.clip(x + noise * rng_local.random(n_features).astype(np.float32), 0.0, 1.0)
        yield x.astype(np.float32), y

def evaluate_snapshot(L1, L2, FB, Xbuf, ybuf, gamma_pair=(0.15,0.10),
                      t_min_in=5, t_max_in=100, jitter=3.5, drop=0.08):
    gamma1, gamma2 = gamma_pair
    H = W = 16
    correct = 0
    for i in range(len(Xbuf)):
        t_grid = latency_code(Xbuf[i], t_min=t_min_in, t_max=t_max_in, jitter_std=jitter, drop_p=drop).reshape(H, W)
        act = L1.forward_no_learn(t_grid); act = lateral_inhibition(act, gamma=gamma1)
        times1, T1 = L1.activations_to_ttfs(act, 5, 90)
        c1, _, _, _ = L2.simulate_episode(times1, T1)
        actm = FB.modulate(act, cls_pred=c1); actm = lateral_inhibition(actm, gamma=gamma2)
        times2, T2 = L1.activations_to_ttfs(actm, 5, 90)
        c, _, _, _ = L2.simulate_episode(times2, T2)
        correct += int(c == ybuf[i])
    return correct / len(Xbuf)

# =========================
# Main ONLINE con controles adaptativos
# =========================
def main_online():
    H = W = 16; n_classes = 10
    n_filters_L1 = 16; m_per_class = 12
    beta_fb = 0.60; eta_f = 3e-4; gamma_pair = (0.15, 0.10)
    t_min_in, t_max_in = 5, 100

    # Módulos
    L1 = ConvSTDPKWTA((H,W), 5, 2, n_filters_L1, 1, 0.02, 0.6, 0.015, 0.012, t_min_in, t_max_in, 0.02, 1.0)
    Oh, Ow = L1.out_h, L1.out_w; n_inputs_L2 = L1.F * Oh * Ow
    kernel = PSPKernel(15e-3, 5e-3, 1e-3, 0.15)
    L2 = RSTDPDecisionPop(n_inputs_L2, n_classes, m_per_class, kernel,
                          v_th=1.0, eta=1.5e-3, a_plus=1.0, a_minus=1.0,
                          tau_pre=20e-3, tau_post=20e-3, lam=0.99, leak=0.98,
                          wmin=0.02, wmax=0.8, dt=1e-3,
                          homeo_target_rate=1.0/n_classes, homeo_lr=4e-3, syn_scale_lr=6e-5,
                          sfa_tau=800.0, sfa_delta=0.08)
    FB = TopDownFeedback(n_classes, L1.F, Oh, Ow, eta_f=eta_f, beta=beta_fb, fmin=-0.8, fmax=0.8)

    # Generador y buffer de evaluación
    gen = make_multiclass_generator(H*W, n_classes, 40, 0.6, 0.20, 0.35, seed=11)
    eval_buf = 800
    X_eval = np.zeros((eval_buf, H*W), dtype=np.float32); y_eval = np.zeros(eval_buf, dtype=np.int32)
    for i in range(eval_buf): X_eval[i], y_eval[i] = next(gen)

    # Warm-up L1
    warmup_steps = 3000
    for s in range(warmup_steps):
        x, _ = next(gen)
        t_grid = latency_code(x, t_min=t_min_in, t_max=t_max_in, jitter_std=3.5, drop_p=0.08).reshape(H, W)
        L1.forward_and_learn(t_grid, boost=True)
    L1.learn_enabled = False  # congelamos L1 tras warm-up
    print(f"[warm-up] L1 listo tras {warmup_steps} pasos.\n")

    # Ventanas y monitores
    win = 600
    last_preds = deque(maxlen=win); last_gts = deque(maxlen=win)
    last_margins = deque(maxlen=win)
    class_winners = deque(maxlen=2000)  # para homeostasis monitor
    nudge_mispred = 0; nudge_total = 0

    # Nudge persistente
    nudge_prob = 0.30  # constante (persistente)
    # Controles adaptativos
    eval_every = 1000
    stagnation_check_span = 4000
    recent_eval = deque(maxlen=stagnation_check_span // eval_every + 1)
    # flags de mantenimiento
    maintenance_mode = False
    maintenance_until = 0

    # Logging
    out_csv = Path("phase3_online_Bplus_adaptive_log.csv")
    rows = []
    N_steps = 16000
    gamma1, gamma2 = gamma_pair

    for step in range(1, N_steps+1):
        x, y = next(gen)

        # Forward L1
        t_grid = latency_code(x, t_min=t_min_in, t_max=t_max_in, jitter_std=3.5, drop_p=0.08).reshape(H, W)
        act = L1.forward_and_learn(t_grid) if L1.learn_enabled else L1.forward_no_learn(t_grid)
        act = lateral_inhibition(act, gamma=gamma1)

        # Primer pase a L2 (sin FB)
        times1, T1 = L1.activations_to_ttfs(act, 5, 90)
        c1, t1, _, _ = L2.simulate_episode(times1, T1)

        # Modulación top-down (durante nudge usar y_true con blend α)
        use_nudge = (rng.random() < nudge_prob)
        if use_nudge:
            alpha = 0.75
            # gating por confianza: si conf es baja, aumentar α (más guía)
            conf_tmp = 0.0  # se calculará en el pase 2, aquí aún no sabemos t_win
            act_y = FB.modulate(act, cls_pred=y)
            act_c1 = FB.modulate(act, cls_pred=c1)
            actm = alpha * act_y + (1.0 - alpha) * act_c1
        else:
            actm = FB.modulate(act, cls_pred=c1)

        actm = lateral_inhibition(actm, gamma=gamma2)
        times2, T2 = L1.activations_to_ttfs(actm, 5, 90)

        # Segundo pase (actor) con posible teacher flag
        pred, t_win, Ttot, used_nudge = L2.simulate_episode(times2, T2, y_true=y, teacher_nudge=use_nudge)

        # Métrica de confianza y margen top-2
        conf = 1.0 - (t_win / float(Ttot))
        margin = L2.last_margin
        last_margins.append(margin)

        # Reward con shaping (latencia + margen top-2)
        lam_conf = 0.5
        k_margin = 0.5
        R = (+1.0 if pred == y else -1.0)
        R *= (1.0 + lam_conf * conf) * (1.0 + k_margin * (1.0 - margin))

        # Actor-only R-STDP
        L2.apply_reward_actor_only(R, pred_class=pred)

        # Hebb supervisado (si hubo nudge)
        if used_nudge:
            L2.apply_supervised_nudge(y_true=y, eta_sup=5e-4, cap_norm=0.4)
            nudge_total += 1
            if pred != y: nudge_mispred += 1

        # Actualizar feedback
        FB.update(actm, cls_pred=pred, R=R)

        # Distribución de ganadores
        class_winners.append(pred)

        # Ventana deslizante de accuracy online
        last_preds.append(pred); last_gts.append(y)
        acc_win = float(np.mean(np.array(last_preds) == np.array(last_gts))) if len(last_preds)==win else None

        # Anti-saturación FB: si ||Fw||∞ se acerca a 0.8, bajar suavemente beta y/o eta_f
        fw_inf = FB.max_inf_norm()
        if fw_inf > 0.75:
            FB.beta = np.float32(max(0.4, float(FB.beta)*0.98))
            FB.eta_f = np.float32(max(1e-4, float(FB.eta_f)*0.98))

        # Mantenimiento/adaptación:
        # 1) Recalentamiento de L1 si estancamiento prolongado o arranque muy bajo
        if step == 2000:
            if acc_win is not None and acc_win < 0.12:
                maintenance_mode = True
                maintenance_until = step + 1000
                L1.learn_enabled = True  # re-enable STDP breve
        if maintenance_mode and step >= maintenance_until:
            L1.learn_enabled = False
            maintenance_mode = False

        # 2) Estancamiento por snapshot: si en 4k pasos la mejora < 0.03 y acc_eval < 0.60, micro-tune L1 1k pasos
        if step % eval_every == 0:
            acc_eval = evaluate_snapshot(L1, L2, FB, X_eval, y_eval, gamma_pair=gamma_pair,
                                         t_min_in=t_min_in, t_max_in=t_max_in, jitter=3.5, drop=0.08)
            recent_eval.append(acc_eval)
            # Monitores adicionales
            vmu = float(np.mean(L2.v_th)); vsd = float(np.std(L2.v_th))
            e_pred = L2.log_e_norm_pred; e_nonpred = L2.log_e_norm_nonpred
            nudge_rate = nudge_mispred / max(1, nudge_total) if nudge_total>0 else 0.0
            margin_avg = float(np.mean(last_margins)) if len(last_margins)>0 else 0.0
            # distribución de ganadores
            cnt = Counter(class_winners)
            dist = [cnt.get(c,0)/max(1,len(class_winners)) for c in range(n_classes)]
            dist_str = ";".join(f"{p:.2f}" for p in dist)

            print(f"[step={step}] Acc_eval={acc_eval:.3f} | Acc_win({win})={(0.0 if acc_win is None else acc_win):.3f} "
                  f"| LR_L2={float(L2.eta):.4f} LR_FB={float(FB.eta_f):.5f} | v_th(mean)={vmu:.3f}±{vsd:.3f} "
                  f"| nudge_p~{nudge_prob:.3f} | nudges={L2.n_nudges} | nudge_misp%={100.0*nudge_rate:.1f} "
                  f"| ||e_pred||={e_pred:.3f} ||e_nonpred||={e_nonpred:.3f} | marginTop2~{margin_avg:.3f} "
                  f"| ||Fw||∞={fw_inf:.3f} | winners={dist_str} | L1_learn={int(L1.learn_enabled)}")

            rows.append([step, (0.0 if acc_win is None else acc_win), acc_eval,
                         float(L2.eta), float(FB.eta_f), vmu, vsd, nudge_prob, L2.n_nudges,
                         e_pred, e_nonpred, margin_avg, fw_inf, int(L1.learn_enabled)] + dist)

            # Trigger de estancamiento
            if len(recent_eval) >= recent_eval.maxlen:
                if (recent_eval[-1] - recent_eval[0] < 0.03) and (recent_eval[-1] < 0.60) and (not maintenance_mode):
                    maintenance_mode = True
                    maintenance_until = step + 1000
                    L1.learn_enabled = True  # micro-tune STDP en L1

        # Decays suaves por estabilidad
        if step % 4000 == 0:
            L2.eta *= np.float32(0.9)
            FB.eta_f *= np.float32(0.95)

        # Protección: si una clase domina >40% en ventana larga, aumentar homeostasis temporalmente
        if step % 2000 == 0 and len(class_winners) >= 1000:
            cnt = Counter(class_winners)
            freqs = np.array([cnt.get(c,0)/len(class_winners) for c in range(n_classes)], dtype=np.float32)
            if np.max(freqs) > 0.40:
                L2.homeo_lr = np.float32(min(8e-3, float(L2.homeo_lr)*1.2))
            else:
                L2.homeo_lr = np.float32(max(4e-3, float(L2.homeo_lr)*0.98))

    # Guardar CSV
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["step","acc_win","acc_eval","lr_l2","lr_fb","vth_mean","vth_std","nudge_p",
                  "n_nudges","e_norm_pred","e_norm_nonpred","marginTop2_avg","Fw_inf","L1_learn"]
        header += [f"win_cls{c}" for c in range(n_classes)]
        w.writerow(header)
        w.writerows(rows)
    print(f"\nLog guardado en {out_csv.resolve()}")

if __name__ == "__main__":
    main_online()
