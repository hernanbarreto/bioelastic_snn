# bioNet_online_B_v2_3a_dynsmooth_adapt_ratio_v2_clean_fixed.py
# Limpieza y correcciones integrales:
# - Declaraciones faltantes agregadas (aw_hist, RESTORE_BEST_AT_END, best_step, burn_in, L1_CONF_THR, L1_AW_THR, etc.).
# - Bloques con indentación incorrecta corregidos.
# - Implementado early-stop opcional (USE_HARD_STOP/HARD_STOP_PATIENCE_K).
# - Implementado "restore best at end" real (con pesos y v_th).
# - Autopilot de nudge y corte dinámico de L1 por confianza/aw.
# - Plateau detector -> FREEZE de FB + decay suave de LR_FB/L2.
# - lateral_inhibition ahora usa ksize de verdad.
# - Removidas variables no usadas; sólo quedan las necesarias.
#
# Nota: El bucle principal corta por max_steps (por defecto 150_000) o early-stop si se activa.

import numpy as np
from collections import deque
from pathlib import Path
import copy

rng = np.random.default_rng(7)

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def acc_from_norm(norm_thr, C):
    base = 1.0 / float(C)
    return float(base + norm_thr * (1.0 - base))

def conv_out_hw(H, W, K=5, S=2):
    Oh = (H - K)//S + 1
    Ow = (W - K)//S + 1
    return Oh, Ow

def dynamic_config(n_classes: int, H=16, W=16, K=5, S=2):
    C = max(2, int(n_classes))
    r = C / 10.0

    F = int(_clamp(round(16 * (r ** 0.5)), 16, 64))
    warmup_steps  = int(_clamp(round(3500 * (r ** 0.5)), 3000, 8000))

    Oh, Ow = conv_out_hw(H, W, K, S)
    N_in = F * Oh * Ow
    ratio_target = 0.2083333333333
    M_float = ratio_target * (N_in / float(C))
    M = int(_clamp(round(M_float), 8, 24))

    nudge_floor        = float(_clamp(0.10 * max(0.0, r - 1.0), 0.0, 0.15))
    nudge_decay_every  = int(_clamp(round(4000 * (r ** 0.5)), 2000, 8000))

    NORM_DECAY  = 29.0/30.0
    NORM_TARGET = 44.0/45.0
    ACC_LR_DECAY_THRESH = acc_from_norm(NORM_DECAY,  C)
    acc_target          = acc_from_norm(NORM_TARGET, C)
    aw_target           = 0.960

    # Boost de L1 dinámico (más largo con más clases)
    boost_len = int(_clamp(round(1500 * (r ** 0.5)), 1200, 3000))

    # Extensión de L1-learn
    L1_learn_extra = int(_clamp(round(1200 * (r ** 0.7)), 800, 2500))

    return dict(
        n_filters_L1=F, warmup_steps=warmup_steps, m_per_class=M,
        nudge_floor=nudge_floor, nudge_decay_every=nudge_decay_every,
        ACC_LR_DECAY_THRESH=ACC_LR_DECAY_THRESH, acc_target=acc_target, aw_target=aw_target,
        Oh=Oh, Ow=Ow, N_in=N_in, L1_learn_extra=L1_learn_extra, ratio_target=ratio_target,
        boost_len=boost_len
    )

# ---------- TTFS / PSP ----------
class PSPKernel:
    def __init__(self, tau=15e-3, tau_s=5e-3, dt=1e-3, T=0.15):
        self.dt = np.float32(dt)
        self.t = np.arange(0.0, T + 1e-12, dt, dtype=np.float32)
        tau = np.float32(tau); tau_s = np.float32(tau_s)
        k = np.exp(-self.t/tau) - np.exp(-self.t/tau_s)
        k[k < 0.0] = 0.0
        if k.max() > 0:
            k = k / k.max()
        self.k = k.astype(np.float32)

def latency_code(x, t_min=5, t_max=100, jitter_std=2.0, drop_p=0.04):
    t_idx = -np.ones_like(x, dtype=np.int32)
    m = x > 0
    if m.any():
        t = t_min + (1.0 - x[m]) * (t_max - t_min)
        if jitter_std > 0: t += rng.normal(0.0, jitter_std, size=t.shape).astype(np.float32)
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

# ---------- ConvSTDPKWTA ----------
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

def lateral_inhibition(act_maps, ksize=3, gamma=0.15):
    """Lateral inhibition con ventana (2r+1)x(2r+1) usando kernel uniforme.
    ksize debe ser impar (3,5,7,...). Si se pasa par, se ajusta al impar más cercano.
    """
    F, H, W = act_maps.shape
    out = act_maps.copy()
    r = int(max(1, (ksize // 2)))
    if (ksize % 2) == 0:
        r = max(1, r)
        ksize = 2*r + 1
    pad = r
    for f in range(F):
        A = act_maps[f]
        Ap = np.pad(A, pad, mode='edge')
        neigh = np.zeros_like(A, dtype=np.float32)
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                neigh += Ap[pad+dy:pad+dy+H, pad+dx:pad+dx+W]
        neigh = neigh / float(ksize * ksize)
        out[f] = np.maximum(0.0, A - gamma*neigh)
    return out


def l2_normalize_rows(W, target=1.0):
    nrm = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    return W * np.minimum(1.0, target / nrm)

# ---------- RSTDPDecisionPop ----------
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

        self.reset_episode_states()
        self.last_x_pre = np.zeros(self.N, dtype=np.float32)
        self.n_nudges = 0
        self.log_e_norm_pred = 0.0
        self.log_e_norm_nonpred = 0.0

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
        self.reset_episode_states()
        self.sfa *= (1.0 - 1.0 / self.sfa_tau)
        Vin = self._build_Vin(times_per_input, T_steps)

        winner_found = False; pred_class = 0; t_win = T_steps
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

            if np.any(post_mask): self.x_post[post_mask] += 1.0
            if np.any(post_mask):
                self.e += self.a_plus * (post_mask.astype(np.float32)[:, None] * self.x_pre[None, :])
            if np.any(pre_mask):
                self.e -= self.a_minus * (self.x_post[:, None] * pre_mask.astype(np.float32)[None, :])

        if not winner_found:
            j = int(np.argmax(self.Vmem)); pred_class = j // self.M; t_win = T_steps
            self.sfa[j] += float(self.sfa_delta) * 0.5

        self.last_x_pre = self.x_pre.copy()
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

        self.win_counts[pred_class] += 1; self.win_total += 1
        if self.win_total >= self.win:
            rate = self.win_counts / float(self.win_total + 1e-8)
            self.v_th += self.homeo_lr * (rate - self.homeo_target)
            self.v_th = np.clip(self.v_th, 0.7*self.v_th_base, 1.6*self.v_th_base)
            self.win_counts[:] = 0; self.win_total = 0

    def apply_supervised_nudge(self, y_true, eta_sup=3e-4):
        if y_true is None: return
        idx_pop = np.arange(y_true*self.M, (y_true+1)*self.M, dtype=np.int32)
        self.W[idx_pop, :] += eta_sup * self.last_x_pre[None, :]
        self.W = np.clip(self.W, float(self.wmin), float(self.wmax))
        self.W[idx_pop, :] = l2_normalize_rows(self.W[idx_pop, :], 1.0)

# ---------- TopDownFeedback ----------
class TopDownFeedback:
    def __init__(self, n_classes, F_maps, Oh, Ow, eta_f=2e-4, fmin=-0.8, fmax=0.8, beta=0.40):
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
        if a_max <= a_min + 1e-8:
            pre = np.zeros_like(A, dtype=np.float32)
        else:
            pre = (A - a_min) / (a_max - a_min + 1e-8)
        dF = self.eta_f * R * pre
        self.Fw[cls_pred] = np.clip(self.Fw[cls_pred] + dF, float(self.fmin), float(self.fmax))
        for f in range(self.F):
            m = np.max(np.abs(self.Fw[cls_pred, f])) + 1e-8
            if m > 0.8:
                self.Fw[cls_pred, f] *= (0.8 / m)

# ---------- GENERADOR Y EVALUACIÓN ----------
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
                      t_min_in=5, t_max_in=100, jitter=2.0, drop=0.04, use_fb=True):
    gamma1, gamma2 = gamma_pair
    H = W = 16
    correct = 0
    for i in range(len(Xbuf)):
        t_grid = latency_code(Xbuf[i], t_min=t_min_in, t_max=t_max_in,
                              jitter_std=jitter, drop_p=drop).reshape(H, W)
        act = L1.forward_no_learn(t_grid); act = lateral_inhibition(act, gamma=gamma1)
        times1, T1 = L1.activations_to_ttfs(act, 5, 90)
        c1, _, _, _ = L2.simulate_episode(times1, T1)
        if use_fb:
            actm = FB.modulate(act, cls_pred=c1)
        else:
            actm = act  # FB inactivo en eval: no modula
        actm = lateral_inhibition(actm, gamma=gamma2)
        times2, T2 = L1.activations_to_ttfs(actm, 5, 90)
        c, _, _, _ = L2.simulate_episode(times2, T2)
        correct += int(c == ybuf[i])
    return correct / len(Xbuf)

# ---------- Main ONLINE ----------
def main_online():
    H = W = 16; n_classes = 5  #para 5/10/20
    cfg = dynamic_config(n_classes, H, W)

    n_filters_L1 = cfg['n_filters_L1']
    m_per_class  = cfg['m_per_class']

    beta_fb_init = 0.40
    eta_f = 2e-4
    gamma_pair = (0.15, 0.10)
    gamma1, gamma2 = gamma_pair
    t_min_in, t_max_in = 5, 100

    print(f"[config] C={n_classes} | L1.filters={n_filters_L1} | warmup={cfg['warmup_steps']} | m_per_class={m_per_class} | nudge_floor={cfg['nudge_floor']:.3f} | nudge_decay_every={cfg['nudge_decay_every']} | ACC_LR_DECAY_THRESH={cfg['ACC_LR_DECAY_THRESH']:.3f} | acc_target={cfg['acc_target']:.3f} | ratio_target={cfg['ratio_target']:.3f} | N_in={cfg['N_in']}")

    L1 = ConvSTDPKWTA((H,W), 5, 2, n_filters_L1, 1, 0.02, 0.6, 0.015, 0.012, t_min_in, t_max_in, 0.02, 1.0)
    Oh, Ow = L1.out_h, L1.out_w; n_inputs_L2 = L1.F * Oh * Ow
    kernel = PSPKernel(15e-3, 5e-3, 1e-3, 0.15)
    L2 = RSTDPDecisionPop(n_inputs_L2, n_classes, m_per_class, kernel,
                          v_th=1.0, eta=1.5e-3, a_plus=1.0, a_minus=1.0,
                          tau_pre=20e-3, tau_post=20e-3, lam=0.99, leak=0.98,
                          wmin=0.02, wmax=0.8, dt=1e-3,
                          homeo_target_rate=1.0/n_classes, homeo_lr=4e-3, syn_scale_lr=6e-5,
                          sfa_tau=800.0, sfa_delta=0.08)
    FB = TopDownFeedback(n_classes, L1.F, Oh, Ow, eta_f=eta_f, beta=beta_fb_init, fmin=-0.8, fmax=0.8)

    L2.eta0 = np.float32(L2.eta)

    gen = make_multiclass_generator(H*W, n_classes, 40, 0.6, 0.20, 0.35, seed=11)

    eval_buf = 800
    X_eval = np.zeros((eval_buf, H*W), dtype=np.float32); y_eval = np.zeros(eval_buf, dtype=np.int32)
    for i in range(eval_buf): X_eval[i], y_eval[i] = next(gen)

    warmup_steps = cfg['warmup_steps']
    for s in range(warmup_steps):
        x, _ = next(gen)
        t_grid = latency_code(x, t_min=t_min_in, t_max=t_max_in, jitter_std=2.0, drop_p=0.04).reshape(H, W)
        boost = (s < min(cfg['boost_len'], warmup_steps//2))
        L1.forward_and_learn(t_grid, boost=boost)
    print(f"[warm-up] L1 listo tras {warmup_steps} pasos.\n")

    win = 600
    last_preds = deque(maxlen=win); last_gts = deque(maxlen=win)
    last_winners = deque(maxlen=win)

    learn_enabled = True

    # -------- Estado/cfg extra (declaraciones necesarias) --------
    fb_cooldown = 0
    best_weights_L1 = None
    best_weights_L2 = None
    best_weights_FB = None

    burn_in = 0  # no burn-in por defecto
    aw_hist = deque(maxlen=12)  # para plateau detection
    l1_auto_cut = False

    # thresholds de corte temprano de L1 (dependen de C)
    L1_CONF_THR = 0.075
    L1_AW_THR   = (0.62 if n_classes <= 10 else 0.52)

    # tracking del mejor y restore
    RESTORE_BEST_AT_END = True
    best_step = -1

    # early-stop (opcional)
    USE_HARD_STOP = False
    HARD_STOP_PATIENCE_K = 8
    no_improve_k = 0

    ckpt_dir = Path("./ckpts_v2_3a"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    def save_ckpt(tag):
        # usa la variable "step" del scope de main_online
        np.savez_compressed(ckpt_dir / f"ckpt_{tag}.npz",
                            L1_w=L1.w, L2_W=L2.W, FB_Fw=FB.Fw, L2_vth=L2.v_th,
                            step=step)

    acc_ema = None; best_ema = -1.0; ema_alpha = 0.3; min_delta = 0.002
    acc_target = cfg['acc_target']; aw_target = cfg['aw_target']; target_hits = 3
    over_target_k = 0; aw_over_k = 0
    ACC_LR_DECAY_THRESH = cfg['ACC_LR_DECAY_THRESH']
    RESUME_DROP = 0.010; RESUME_FLOOR = acc_from_norm(0.70, n_classes)

    L2_ETA0_MIN, L2_ETA0_MAX = 3e-5, 1.5e-3
    FB_ETA_MIN, FB_ETA_MAX   = 2e-5, 2e-4

    nudge_p0 = 0.30
    nudge_decay_every = cfg['nudge_decay_every']
    nudge_decay = 0.5
    nudge_floor = cfg['nudge_floor']
    stuck_boost_until = 6000

    # Gating de FB
    FB_FREEZE_MAX_STEPS = warmup_steps + 1000
    FB_AW_FLOOR = 0.10

    base_homeo_lr = float(L2.homeo_lr)

    best_weights_L1 = copy.deepcopy(L1.w)
    best_weights_L2 = copy.deepcopy(L2.W)
    best_weights_FB = copy.deepcopy(FB.Fw)
    best_v_th_L2 = copy.deepcopy(L2.v_th)
    nudge_used = 0; nudge_misp = 0

    lam_min, lam_max = 0.55, 0.95
    conf_ema = 0.0; conf_alpha = 0.20

    # control de duración
    max_steps = 150_000

    step = 0; training_active = True
    while training_active:
        step += 1
        if step > max_steps:
            training_active = False
            break

        base = 1.0 / n_classes
        q = 0.0 if (acc_ema is None) else max(0.0, (acc_ema - base) / (1.0 - base))

        p_sched = nudge_p0 * (nudge_decay ** ((step-1)//nudge_decay_every))
        p_prog  = nudge_p0 * (1.0 - 0.7*q)
        p = max(nudge_floor, min(p_prog, p_sched))

        x, y = next(gen)
        L2.homeo_lr = np.float32(base_homeo_lr * (2.0 if step <= 8000 else 1.0))

        t_grid = latency_code(x, t_min=t_min_in, t_max=t_max_in, jitter_std=2.0, drop_p=0.04).reshape(H, W)

        L1_learn_horizon = max(4000, int(0.7*warmup_steps)) + cfg['L1_learn_extra']
        do_l1_learn = (step <= L1_learn_horizon) and (not l1_auto_cut) and (learn_enabled or (burn_in > 0))
        if do_l1_learn:
            act = L1.forward_and_learn(t_grid, boost=False); L1_learn_flag = 1
        else:
            act = L1.forward_no_learn(t_grid); L1_learn_flag = 0

        act = lateral_inhibition(act, gamma=gamma1)

        times1, T1 = L1.activations_to_ttfs(act, 5, 90)
        c1, t1, _, _ = L2.simulate_episode(times1, T1)

        # --- Gating FB: decidir si FB está activo ANTES de modular ---
        aw_tmp = 0.0 if len(last_preds)<win else float(np.mean(np.array(last_preds) == np.array(last_gts)))
        freeze_by_step = (step < FB_FREEZE_MAX_STEPS)
        freeze_by_aw   = (aw_tmp < FB_AW_FLOOR)
        fb_active = (not freeze_by_step) and (not freeze_by_aw)

        if fb_active:
            actm = FB.modulate(act, cls_pred=c1)
        else:
            actm = act  # FB inactivo: no modula

        actm = lateral_inhibition(actm, gamma=gamma2)

        times2, T2 = L1.activations_to_ttfs(actm, 5, 90)

        use_nudge = (step <= 12000) and (rng.random() < p)
        acc_win = 0.0 if len(last_preds)<win else float(np.mean(np.array(last_preds) == np.array(last_gts)))
        if step <= stuck_boost_until and acc_win < 0.18:
            if rng.random() < 0.25:
                use_nudge = True

        pred, t_win, Ttot, used_nudge_flag = L2.simulate_episode(times2, T2, y_true=y, teacher_nudge=use_nudge)

        if used_nudge_flag:
            nudge_used += 1
            if pred != y: nudge_misp += 1

        conf_inst = 1.0 - (t_win / float(Ttot))
        conf_ema = conf_alpha * conf_inst + (1.0 - conf_alpha) * conf_ema
        lam = lam_min + (lam_max - lam_min) * (0.7 * (q ** 0.7) + 0.3 * conf_ema)
        lam = float(np.clip(lam, lam_min, lam_max))

        R = (+1.0 if pred == y else -1.0) * (1.0 + lam * conf_inst)

        safe_scale = 10.0 / (1.0 + L2.log_e_norm_pred)
        L2.eta = np.float32(L2.eta0 * min(1.0, safe_scale))
        if learn_enabled:
            L2.apply_reward_actor_only(R, pred_class=pred)
        L2.eta = np.float32(L2.eta0)

        if learn_enabled and used_nudge_flag:
            ratio_now = (n_classes * m_per_class) / float(n_inputs_L2)
            ratio_base = 0.2083333333333
            mispr = (nudge_misp / max(1, nudge_used))
            scale_sup = np.clip(ratio_now/ratio_base, 0.7, 1.6)
            eta_sup = 3e-4 * (1.0 + 0.5*mispr) * scale_sup
            if acc_win < 0.08:  # arranque muy frío
                eta_sup *= 1.25
            L2.apply_supervised_nudge(y_true=y, eta_sup=eta_sup)

        margin_top2_proxy = float(np.mean(np.abs(FB.Fw)))
        inf_norm_Fw = float(np.max(np.abs(FB.Fw)))
        allow_fb_update = True

        if (not fb_active):
            allow_fb_update = False  # si no está activo, tampoco aprende
        elif fb_cooldown > 0:
            fb_cooldown -= 1
            allow_fb_update = False
        else:
            if (inf_norm_Fw > 0.42) or (margin_top2_proxy > 0.11):
                FB.eta_f *= np.float32(0.8)
                fb_cooldown = 400  # micro-cooldown por saturación
                allow_fb_update = False

        if learn_enabled and allow_fb_update:
            FB.update(actm, cls_pred=pred, R=R)

        last_preds.append(pred); last_gts.append(y); last_winners.append(pred)

        if step % 1000 == 0:
            # En eval usamos FB sólo si ya está activo y AW supera piso
            EVAL_WITH_FB = False
            fb_eval_active = (EVAL_WITH_FB and fb_active and (acc_win >= FB_AW_FLOOR))
            acc_eval = evaluate_snapshot(L1, L2, FB, X_eval, y_eval, gamma_pair=gamma_pair,
                                         t_min_in=t_min_in, t_max_in=t_max_in, jitter=2.0, drop=0.04,
                                         use_fb=fb_eval_active)
            aw = 0.0 if len(last_preds)<win else float(np.mean(np.array(last_preds) == np.array(last_gts)))

            # --- actualizar historial de aw y detectar meseta ---
            aw_hist.append(aw)
            plateau = False
            if len(aw_hist) == aw_hist.maxlen:
                slope = (aw_hist[-1] - aw_hist[0]) / float(aw_hist.maxlen - 1)
                plateau = (slope < 0.0004)
            if plateau and fb_active and (fb_cooldown == 0):
                fb_cooldown = 1000
                FB.eta_f = np.float32(max(FB_ETA_MIN, float(FB.eta_f) * 0.70))
                L2.eta0  = np.float32(max(L2_ETA0_MIN, float(L2.eta0) * 0.90))
                print(f"[FB-FREEZE] Plateau detectado (slope≈{slope:.5f}). Congelo FB 1000 pasos y bajo LR_FB.")

            # --- EMA de confianza (marginTop2 proxy) y corte automático de L1 ---
            conf_ema = conf_alpha * float(margin_top2_proxy) + (1.0 - conf_alpha) * conf_ema
            if (not l1_auto_cut) and step > warmup_steps and ((conf_ema >= L1_CONF_THR) or (aw >= L1_AW_THR)):
                l1_auto_cut = True
                print(f"[L1-OFF] Corte automático de L1 en step {step} (conf_ema={conf_ema:.3f}, aw={aw:.3f}).")

            lr2 = float(L2.eta0); lrfb = float(FB.eta_f)
            vmu = float(np.mean(L2.v_th)); vsd = float(np.std(L2.v_th))
            winners_rate = np.zeros(n_classes, dtype=np.float32)
            if len(last_winners) > 0:
                lw = np.array(last_winners, dtype=np.int32)
                for c in range(n_classes):
                    winners_rate[c] = np.mean(lw == c)
            winners_str = ";".join([f"{r:.02f}" for r in winners_rate.tolist()])
            nudge_misp_pct = (100.0 * nudge_misp / max(1, nudge_used)) if nudge_used>0 else 0.0

            # --- Autopilot de nudge_p para sostener tasa de 'mispred' objetivo ---
            target_misp = (0.78 if n_classes <= 10 else 0.88)
            err_gap = (nudge_misp_pct / 100.0) - target_misp
            if abs(err_gap) > 0.02:
                new_nudge_p0 = nudge_p0 * (1.0 + 0.5 * err_gap)
                nudge_p0 = float(np.clip(new_nudge_p0, nudge_floor, 0.30))

            print(
                f"[step={step}] Acc_eval={acc_eval:.3f} | Acc_win({win})={aw:.3f} "
                f"| LR_L2={lr2:.4f} LR_FB={lrfb:.5f} | v_th(mean)={vmu:.3f}±{vsd:.3f} "
                f"| nudge_p~{p:.3f} | nudges={L2.n_nudges} | nudge_misp%={nudge_misp_pct:.1f} "
                f"| ||e_pred||={L2.log_e_norm_pred:.3f} ||e_nonpred||={L2.log_e_norm_nonpred:.3f} "
                f"| marginTop2~{margin_top2_proxy:.3f} | ||Fw||∞={inf_norm_Fw:.3f} "
                f"| winners={winners_str} | L1_learn={L1_learn_flag} | FB_active={int(fb_active)}"
            )

            # EMA y best tracking
            acc_ema = acc_eval if acc_ema is None else (ema_alpha*acc_eval + (1-ema_alpha)*acc_ema)
            if best_ema < 0 or (acc_ema > best_ema + min_delta):
                best_ema = acc_ema
                best_step = step
                best_weights_L1 = copy.deepcopy(L1.w)
                best_weights_L2 = copy.deepcopy(L2.W)
                best_weights_FB = copy.deepcopy(FB.Fw)
                best_v_th_L2    = copy.deepcopy(L2.v_th)
                save_ckpt(f"best_{step}")
                no_improve_k = 0
            else:
                no_improve_k += 1

            # Early-stop opcional
            if USE_HARD_STOP and (no_improve_k >= HARD_STOP_PATIENCE_K):
                print(f"[EARLY-STOP] Sin mejora en {no_improve_k} evals. Corto entrenamiento.")
                training_active = False

            # LR decay suave cuando se aproxima al target
            if acc_eval >= ACC_LR_DECAY_THRESH:
                x0, x1 = ACC_LR_DECAY_THRESH, acc_target
                y0, y1 = 0.985, 0.960
                t = 0.0 if x1 <= x0 else float(np.clip((acc_eval - x0) / (x1 - x0), 0.0, 1.0))
                cool = y0 + t * (y1 - y0)
                L2.eta0 = np.float32(np.clip(float(L2.eta0) * cool, L2_ETA0_MIN, L2_ETA0_MAX))
                FB.eta_f = np.float32(np.clip(float(FB.eta_f) * (cool ** 1.05), FB_ETA_MIN, FB_ETA_MAX))

            # pausa de aprendizaje si hay desempeño sostenido
            over_target_k = (over_target_k + 1) if acc_eval >= acc_target else 0
            aw_over_k     = (aw_over_k + 1) if (aw >= aw_target) else 0
            if learn_enabled and (over_target_k >= target_hits) and (aw_over_k >= target_hits):
                learn_enabled = False
                save_ckpt(f"pause_{step}")
                print(f"[PAUSE] Alto desempeño sostenido: acc_eval={acc_eval:.3f}, aw={aw:.3f}. Pauso L2/FB.")

            # reanudar si cae el desempeño respecto del mejor
            if (not learn_enabled) and ((best_ema - acc_eval) >= RESUME_DROP or acc_eval < RESUME_FLOOR):
                learn_enabled = True
                L2.eta0 = np.float32(np.clip(float(L2.eta0) * 1.05, L2_ETA0_MIN, L2_ETA0_MAX))
                FB.eta_f = np.float32(np.clip(float(FB.eta_f) * 1.05, FB_ETA_MIN, FB_ETA_MAX))
                print(f"[RESUME] Desempeño cayó (acc_eval={acc_eval:.3f}). Reanudo aprendizaje con LR suave.")

    # ---- Restore final del mejor estado (opcional) ----
    if RESTORE_BEST_AT_END and (best_ema > 0):
        if best_weights_L1 is not None:
            L1.w[...] = best_weights_L1
        if best_weights_L2 is not None:
            L2.W[...] = best_weights_L2
        if best_weights_FB is not None:
            FB.Fw[...] = best_weights_FB
        if 'best_v_th_L2' in locals() and best_v_th_L2 is not None:
            L2.v_th[...] = best_v_th_L2
        if best_step >= 0:
            print(f"[RESTORE] Se restauró el mejor estado (step={best_step}, Acc_eval≈{best_ema:.3f}).")

if __name__ == "__main__":
    main_online()
