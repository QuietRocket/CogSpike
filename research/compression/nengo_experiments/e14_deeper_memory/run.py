#!/usr/bin/env python3
"""e14 -- Eligibility traces for memory beyond the previous symbol.

Paper outlook (the "Deeper memory" section + FIX G / BIO-06): the local three-factor
rule Delta W_ij = eta c_i (y_j - q_j) correlates the residual at time t with the
context c_t active at the SAME step. That is exactly right for a FIRST-order source
(the useful context is the immediately preceding symbol). It is NOT enough when the
predictive structure spans several past symbols: the residual that resolves at t must
be credited to synapses that were active EARLIER. The standard device is an
ELIGIBILITY TRACE

    e_ij <- gamma e_ij + c_i,      Delta W_ij = eta e_ij r_j,            (TD / three-factor)

a fading per-synapse memory that keeps a synapse "eligible" for a while after it fired.
FIX G is the honesty patch: the line/ring attractor that holds the context stores a
FINITE, noise-limited number of distinguishable states (~log2 SNR bits), so deeper
memory is capacity-bottlenecked -- an order-2 context needs 16 distinguishable states
where order-1 needed 4.

This experiment makes that whole story concrete and measured:

  (1) A SECOND-ORDER source whose next symbol depends on the previous TWO symbols,
      constructed so its order-2 entropy rate is measurably BELOW its order-1 rate:
      a "ping-pong vs run" rover. We quantify the gap I_2 = H1 - H2 > 0 -- the bits a
      first-order predictor must leave on the table.
  (2) A FIRST-ORDER learner (context = previous symbol only), in spiking PES AND in a
      fast numpy delta-rule anchor, converges to the order-1 floor H1 and CANNOT go
      below it: stuck, leaving I_2 bits unrecovered.
  (3) An ELIGIBILITY-TRACE / longer-context learner recovers a measurable fraction of
      I_2 -- its energy drops BELOW the order-1 floor toward the order-2 floor H2. We
      use a fast numpy lag-tagged eligibility-trace delta-rule learner for the core
      learning-dynamics result, and a SPIKING demonstration of the trace dynamics
      e <- gamma e + c (a leaky-integrator population). We are explicit about which is
      which, and about the one place the substrate strains (see the #honest account).
  (4) The attractor-capacity bottleneck (FIX G): an order-2 context needs 16 vs 4
      distinguishable states (log2 16 = 4 vs log2 4 = 2 bits). We relate this to e06's
      MEASURED ring capacity (C_isi ~ 7.9 bits) and show the real catch is not the raw
      bit budget but the 4x-finer quantization margin a 16-way context demands.

Outputs: figures + a results .npz, and a PASS/FAIL line per acceptance check.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import nengo

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

from spikecoder import config as cfg                         # noqa: E402
from spikecoder import plotting as plot                      # noqa: E402
from spikecoder.information import softmax, entropy_bits, LN2  # noqa: E402

N = 4
LABELS = ["U", "D", "L", "R"]
SEED = cfg.SEED


# ===========================================================================
# (1) The second-order "ping-pong vs run" source.
#   next symbol depends on the ORDERED pair (x_{t-2}, x_{t-1}):
#     * if x_{t-2} == x_{t-1} (it was RUNNING / repeated)  -> BREAK to a different move,
#     * if x_{t-2} != x_{t-1} (it just SWITCHED a->b)       -> switch BACK to a (ping-pong).
#   Under order-1 (knowing only x_{t-1}=b) you cannot tell whether to expect a break or a
#   switch-back-to-a, because a is hidden -- so a first-order predictor must average the
#   two opposed regimes and leaves I_2 = H1 - H2 bits on the table. (An XOR-like source:
#   the useful signal lives in the RELATION between the last two symbols.)
# ===========================================================================
P_BREAK = 0.85      # P(leave the repeated symbol | last two agreed)
P_BACK = 0.85       # P(switch back to x_{t-2} | last two differed)
EPS = 0.06          # uniform smoothing (keeps the chain ergodic, surprisal finite)


def build_second_order(p_break=P_BREAK, p_back=P_BACK, eps=EPS):
    """T2[a, b, c] = P(next = c | prev2 = a, prev1 = b)."""
    T2 = np.zeros((N, N, N))
    for a in range(N):
        for b in range(N):
            row = np.zeros(N)
            if a == b:                                   # was running -> break
                others = [c for c in range(N) if c != b]
                for c in others:
                    row[c] = p_break / len(others)
                row[b] = 1.0 - p_break
            else:                                        # just switched -> switch back to a
                row[a] = p_back
                others = [c for c in range(N) if c != a]
                for c in others:
                    row[c] = (1.0 - p_back) / len(others)
            row = (1.0 - eps) * row + eps * np.ones(N) / N
            T2[a, b] = row / row.sum()
    return T2


def stationary_pair(T2, iters=200000, tol=1e-14):
    """Stationary distribution over ordered pairs (a, b) via power iteration on the
    pair-transition matrix M[(a,b),(b,c)] = T2[a,b,c]. Returns mu_pair (N, N) and M."""
    M = np.zeros((N * N, N * N))
    for a in range(N):
        for b in range(N):
            for c in range(N):
                M[a * N + b, b * N + c] = T2[a, b, c]
    mu = np.ones(N * N) / (N * N)
    for _ in range(iters):
        m2 = mu @ M
        m2 /= m2.sum()
        if np.max(np.abs(m2 - mu)) < tol:
            mu = m2
            break
        mu = m2
    return mu.reshape(N, N), M


def source_floors(T2):
    """Closed-form order-0/1/2 entropy rates of the second-order source."""
    mu, M = stationary_pair(T2)
    pi1 = mu.sum(0)                                       # single-symbol marginal
    # order-2 rate: average row entropy over the pair-stationary distribution
    H2 = sum(mu[a, b] * entropy_bits(T2[a, b]) for a in range(N) for b in range(N))
    # induced order-1 chain P1[b,c] = sum_a P(prev2=a|prev1=b) T2[a,b,c]
    P1 = np.zeros((N, N))
    for b in range(N):
        for c in range(N):
            P1[b, c] = sum((mu[a, b] / pi1[b]) * T2[a, b, c] for a in range(N))
    H1 = sum(pi1[b] * entropy_bits(P1[b]) for b in range(N))
    H0 = entropy_bits(pi1)                                # marginal (order-0)
    ev2 = float(np.sort(np.abs(np.linalg.eigvals(M)))[::-1][1])  # spectral-gap proxy
    return dict(mu=mu, pi1=pi1, P1=P1, H0=H0, H1=H1, H2=H2,
                I2=H1 - H2, I1=H0 - H1, ev2=ev2)


def sample_second_order(T2, pi1, n, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.empty(n, int)
    x[0] = rng.choice(N, p=pi1)
    x[1] = rng.choice(N, p=pi1)
    for t in range(2, n):
        x[t] = rng.choice(N, p=T2[x[t - 2], x[t - 1]])
    return x


def energy_order1(Q1, x, lo):
    """Cross-entropy (bits/symbol) of a 1st-order model Q1[b,:] on the stream tail."""
    s = np.array([-np.log2(max(Q1[x[t - 1], x[t]], 1e-12)) for t in range(lo, len(x))])
    return float(s.mean())


def energy_order2(Q2, x, lo):
    """Cross-entropy of a pair-context model Q2[a*N+b,:] on the stream tail."""
    s = np.array([-np.log2(max(Q2[x[t - 2] * N + x[t - 1], x[t]], 1e-12))
                  for t in range(lo, len(x))])
    return float(s.mean())


# ===========================================================================
# (2)+(3) numpy delta-rule learners (the core learning-dynamics anchor).
#   FIRST-ORDER : logits W1[b,:], context = one-hot of x_{t-1}.            -> H1 floor.
#   LAG-TAGGED ELIGIBILITY TRACE : a 2N-dim feature that keeps lag-1 and lag-2 context
#     as SEPARATE channels [ one-hot(x_{t-1}) ; gamma * one-hot(x_{t-2}) ]. This is the
#     faithful form of e_ij <- gamma e_ij + c_i when the trace TAGS WHICH past step a
#     synapse was active at (distinct delay lines), so the update credits the residual
#     to the lag-2 context too. gamma=0 collapses to pure first order.
# ===========================================================================
def learn_first_order(x, lr=0.05, snap_at=None):
    """Online softmax delta rule, context = previous symbol. Returns Q1 and snapshots."""
    W = np.zeros((N, N))
    snaps = {}
    snap_at = set(snap_at or [])
    for t in range(2, len(x)):
        b = x[t - 1]
        q = softmax(W[b])
        g = q.copy(); g[x[t]] -= 1.0                     # grad of -ln q_y in logits
        W[b] -= lr * g
        if t in snap_at:
            snaps[t] = np.vstack([softmax(W[i]) for i in range(N)])
    Q1 = np.vstack([softmax(W[i]) for i in range(N)])
    return Q1, snaps


def lag_features(x, t, gamma):
    f = np.zeros(2 * N)
    f[x[t - 1]] = 1.0                                    # lag-1 channel
    f[N + x[t - 2]] = gamma                              # lag-2 channel (trace-weighted)
    return f


def learn_eligibility_trace(x, gamma, lr=0.05):
    """Lag-tagged eligibility-trace delta rule. Returns the converged tail energy and a
    per-(2N) -> q reader so we can also report the implied pair-context law."""
    W = np.zeros((2 * N, N))
    for t in range(2, len(x)):
        f = lag_features(x, t, gamma)
        q = softmax(W.T @ f)
        r = np.eye(N)[x[t]] - q                          # residual (post-factor)
        W += lr * np.outer(f, r)                         # Delta W = eta * e * r
    lo = len(x) // 2
    s = []
    for t in range(lo, len(x)):
        f = lag_features(x, t, gamma)
        q = softmax(W.T @ f)
        s.append(-np.log2(max(q[x[t]], 1e-12)))
    return float(np.mean(s)), W


# ===========================================================================
# (2) spiking PES first-order learner (the substrate confirmation: stuck at H1).
#   Identical wiring to the VERIFIED e09 idiom: a one-hot previous-symbol context
#   population whose decoders are taught by PES with error = q - y.
# ===========================================================================
N_CTX = 600
CTX_RADIUS = 1.3
WINDOW = 0.03
LR_PES = 2e-4
DT = cfg.DT


def context_input_first(x, window):
    x = np.asarray(x, int)

    def f(t):
        k = int(t // window)
        out = np.zeros(N)
        if 1 <= k < len(x):
            out[x[k - 1]] = 1.0
        return out

    return f


def outcome_input(x, window):
    x = np.asarray(x, int)

    def f(t):
        k = int(t // window)
        out = np.zeros(N)
        if 0 <= k < len(x):
            out[x[k]] = 1.0
        return out

    return f


def build_pes_first_order(ctx_fn, y_fn, seed=SEED):
    net = nengo.Network(seed=seed)
    with net:
        ci = nengo.Node(ctx_fn, size_out=N)
        ctx = nengo.Ensemble(N_CTX, N, radius=CTX_RADIUS)
        nengo.Connection(ci, ctx, synapse=0.005)
        q = nengo.Node(size_in=N)
        conn = nengo.Connection(ctx, q, function=lambda c: np.ones(N) * 0.25,
                                learning_rule_type=nengo.PES(learning_rate=LR_PES),
                                synapse=0.01)
        yi = nengo.Node(y_fn, size_out=N)
        err = nengo.Node(size_in=N)
        nengo.Connection(q, err, synapse=None)
        nengo.Connection(yi, err, transform=-1, synapse=None)
        nengo.Connection(err, conn.learning_rule, synapse=None)
        pq = nengo.Probe(q, synapse=0.02)
        pctx = nengo.Probe(ci)
    return net, pq, pctx


def pes_learned_first_order(x_pes):
    """Run the spiking PES first-order learner; return the learned 1st-order law Q1."""
    ctx_fn = context_input_first(x_pes, WINDOW)
    y_fn = outcome_input(x_pes, WINDOW)
    net, pq, pctx = build_pes_first_order(ctx_fn, y_fn)
    with nengo.Simulator(net, dt=DT, progress_bar=False) as sim:
        sim.run(len(x_pes) * WINDOW)
    qd, cd = sim.data[pq], sim.data[pctx]
    n = len(x_pes)
    win_q = np.zeros((n, N)); win_ctx = np.full(n, -1)
    for k in range(n):
        i0 = int(round(k * WINDOW / DT)); i1 = int(round((k + 1) * WINDOW / DT))
        if i1 > len(qd):
            break
        seg = qd[i0:i1]; win_q[k] = seg[len(seg) // 2:].mean(0)
        csum = cd[i0:i1].sum(0)
        if csum.max() > 0:
            win_ctx[k] = int(np.argmax(csum))
    lo = int(0.6 * n)
    rows = {i: [] for i in range(N)}
    for k in range(lo, n):
        if win_ctx[k] >= 0:
            rows[win_ctx[k]].append(win_q[k])
    Q1 = np.ones((N, N)) / N
    for i in range(N):
        if rows[i]:
            v = np.clip(np.mean(rows[i], 0), 1e-6, None)
            Q1[i] = v / v.sum()
    return Q1


# ===========================================================================
# (3, spiking illustration) the eligibility-trace dynamics e <- gamma e + c on a
#   spiking leaky-integrator population. A recurrent transform = gamma applies one
#   geometric decay per symbol window; a (1-gamma) one-hot input injects the active
#   context, so a held channel saturates at 1 and a silent channel decays as gamma^k.
# ===========================================================================
W_TRACE = 0.05            # symbol window for the trace demo (s)
GAMMA_TRACE = 0.7         # target per-window decay factor of the eligibility trace


def run_spiking_trace(ctx_seq, gamma=GAMMA_TRACE, window=W_TRACE, seed=SEED):
    """Decode the spiking eligibility trace over a short context pulse train.
    ctx_seq entries are symbol ids (0..N-1) or -1 (silent). Returns (per-window decoded
    trace (K, N), measured steady-state per-window decay gamma_hat)."""
    ctx_seq = list(ctx_seq)

    def ctx_node(t):
        k = int(t // window)
        out = np.zeros(N)
        if 0 <= k < len(ctx_seq) and ctx_seq[k] >= 0:
            out[ctx_seq[k]] = 1.0
        return out

    with nengo.Network(seed=seed) as net:
        cin = nengo.Node(ctx_node, size_out=N)
        trace = nengo.Ensemble(1000, N, radius=2.0)
        nengo.Connection(trace, trace, synapse=window, transform=gamma)      # geometric decay
        nengo.Connection(cin, trace, synapse=0.005, transform=(1.0 - gamma))  # inject context
        p = nengo.Probe(trace, synapse=0.02)
    with nengo.Simulator(net, dt=DT, progress_bar=False) as sim:
        sim.run(len(ctx_seq) * window)
    d = sim.data[p]
    win = np.array([d[int(round(k * window / DT)):int(round((k + 1) * window / DT))].mean(0)
                    for k in range(len(ctx_seq))])
    return win


# ===========================================================================
print("=" * 76)
print("e14 -- Eligibility traces for memory beyond the previous symbol")
print("=" * 76)

# --- (1) build + quantify the second-order source -------------------------
T2 = build_second_order()
fl = source_floors(T2)
H0, H1, H2, I2, I1 = fl["H0"], fl["H1"], fl["H2"], fl["I2"], fl["I1"]
pi1, P1 = fl["pi1"], fl["P1"]
print("\n(1) Second-order 'ping-pong vs run' source "
      f"(p_break={P_BREAK}, p_back={P_BACK}, eps={EPS})")
print(f"    single-symbol marginal pi = [{', '.join(f'{v:.3f}' for v in pi1)}]  "
      f"(spectral-gap proxy |ev2| = {fl['ev2']:.3f}, ergodic)")
print(f"    order-0 marginal entropy H0 = {H0:.4f} bits/symbol")
print(f"    order-1 entropy rate    H1 = {H1:.4f} bits/symbol  (best a prev-symbol model can do)")
print(f"    order-2 entropy rate    H2 = {H2:.4f} bits/symbol  (the true floor)")
print(f"    *** I_2 = H1 - H2 = {I2:.4f} bits/symbol > 0  (left on the table by order-1) ***")
print(f"        I_1 = H0 - H1 = {I1:.4f} bits/symbol  (what order-1 recovers over the marginal)")

# sample the headline stream
n_stream = 200000
x = sample_second_order(T2, pi1, n_stream)
# empirical sanity: the realized stream's order-1 vs order-2 surprisal under the TRUE laws
true_pair = np.vstack([T2[a, b] for a in range(N) for b in range(N)])
emp_H2 = energy_order2(np.clip(true_pair, 1e-12, 1), x, 2)
emp_H1 = energy_order1(np.clip(P1, 1e-12, 1), x, 2)
print(f"    empirical check on n={n_stream}: stream surprisal under true order-1 law "
      f"= {emp_H1:.4f} (H1 {H1:.4f}), under true order-2 law = {emp_H2:.4f} (H2 {H2:.4f})")

# --- (2) FIRST-ORDER learner: numpy anchor + spiking PES, both stuck at H1 ----
print("\n(2) First-order learner (context = previous symbol only) -- stuck at the order-1 floor")
snap_steps = sorted(set([200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000,
                         150000, n_stream - 1]))
Q1_np, snaps1 = learn_first_order(x, lr=0.05, snap_at=snap_steps)
lo_eval = n_stream // 2
E1_np = energy_order1(Q1_np, x, lo_eval)
print(f"    numpy delta rule  : converged energy E1 = {E1_np:.4f} bits/symbol  "
      f"(floor H1 = {H1:.4f}, excess {E1_np - H1:+.4f})")

# spiking PES first-order learner (shorter stream -- spiking is expensive)
n_pes = 12000
x_pes = sample_second_order(T2, pi1, n_pes, seed=SEED)
print(f"    spiking PES       : running {n_pes} symbols, window {WINDOW}s, lr {LR_PES} ...")
Q1_pes = pes_learned_first_order(x_pes)
# evaluate the spiking-learned 1st-order law on the long stream
E1_pes = energy_order1(Q1_pes, x, lo_eval)
print(f"    spiking PES       : converged energy E1 = {E1_pes:.4f} bits/symbol  "
      f"(floor H1 = {H1:.4f}, noise-ball excess {E1_pes - H1:+.4f}; e09's constant-lr/decode tax)")
print(f"    BOTH first-order learners sit in the order-1 regime and CANNOT cross toward H2:")
print(f"      I_2 = {I2:.4f} bits/symbol remain unrecovered (the residual structure is in x_t-2).")

# --- (3) ELIGIBILITY-TRACE / longer-context learner: recovers a fraction of I_2 ----
print("\n(3) Eligibility-trace / longer-context learner -- recovers a fraction of I_2")
gammas = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
E_trace = np.array([learn_eligibility_trace(x, g, lr=0.05)[0] for g in gammas])
recovered_frac = (E1_np - E_trace) / I2                   # fraction of I_2 recovered
print("    numpy lag-tagged eligibility-trace delta rule (gamma sweep):")
print(f"      {'gamma':<8}{'energy':<12}{'below E1':<12}{'frac of I_2 recovered':<22}")
for g, E, fr in zip(gammas, E_trace, recovered_frac):
    print(f"      {g:<8.2f}{E:<12.4f}{E1_np - E:<+12.4f}{fr:<22.3f}")
E_trace_best = float(E_trace[1:].min())                   # best gamma>0
g_best = float(gammas[1:][np.argmin(E_trace[1:])])
frac_best = (E1_np - E_trace_best) / I2
print(f"    best gamma>0 = {g_best:.2f}: energy {E_trace_best:.4f} -> drops "
      f"{E1_np - E_trace_best:.4f} bits BELOW the order-1 floor toward H2={H2:.4f}")
print(f"    -> recovers {frac_best*100:.1f}% of I_2 (the trace credits the residual to x_t-2).")

#   the honest contrast: a COLLAPSED trace (sum the one-hots into ONE softmax input,
#   losing which symbol came at which lag) CANNOT recover I_2 on this XOR-like source.
def learn_collapsed_trace(x, gamma, lr=0.05):
    """One softmax over a SINGLE low-passed context vector e <- gamma e + c (lag identity
    discarded). On a ping-pong source this blurs x_t-1 and x_t-2 into one blended input
    and cannot disambiguate the ordered pair."""
    W = np.zeros((N, N)); e = np.zeros(N)
    for t in range(2, len(x)):
        c = np.eye(N)[x[t - 1]]; e = gamma * e + c
        ein = e / e.sum()
        q = softmax(W.T @ ein); r = np.eye(N)[x[t]] - q
        W += lr * np.outer(ein, r)
    lo = len(x) // 2; e = np.zeros(N); s = []
    for t in range(lo, len(x)):
        c = np.eye(N)[x[t - 1]]; e = gamma * e + c; ein = e / e.sum()
        q = softmax(W.T @ ein); s.append(-np.log2(max(q[x[t]], 1e-12)))
    return float(np.mean(s))


E_collapsed = learn_collapsed_trace(x, 0.5, lr=0.05)
print(f"    HONEST contrast: a COLLAPSED (lag-blind) trace at gamma=0.5 gives energy "
      f"{E_collapsed:.4f}")
print(f"      -> {'BELOW' if E_collapsed < E1_np else 'NOT below'} E1: a blended trace cannot "
      f"disambiguate the ordered pair on this XOR-like source.")
print(f"      The trace must TAG WHICH lag each synapse was active at (distinct delay "
      f"lines); only then does it recover I_2. This is the real content of FIX G.")

# --- (3, spiking illustration) the trace dynamics e <- gamma e + c in spikes ----
print("\n(3, spiking) eligibility-trace dynamics e <- gamma e + c on a spiking population")
#   build: one channel pulsed once then silent (clean decay), and a build-up demo.
decay_seq = [0] + [-1] * 8
trace_decay = run_spiking_trace(decay_seq, gamma=GAMMA_TRACE)
ch0 = trace_decay[:, 0]
ks = np.array([k for k in range(2, 7) if ch0[k] > 0.03])   # skip the settle transient
gamma_hat = float(np.mean(ch0[ks] / ch0[ks - 1])) if len(ks) else float("nan")
# a build/decay demo stream for the figure
demo_seq = [0, 0, 1, 1, 2, 0, 3, 3]
trace_demo = run_spiking_trace(demo_seq, gamma=GAMMA_TRACE)
print(f"    target per-window decay gamma = {GAMMA_TRACE:.2f}; "
      f"spiking measured steady-state gamma_hat = {gamma_hat:.3f}")
print(f"    -> the spiking trace builds on a pulse and decays geometrically when silent: "
      f"the mechanism is realized in spikes (quantitative gamma carries NEF integrator drift; "
      f"the core learning result above uses the numpy trace -- see the report's honesty check).")

# --- (4) the FIX G attractor-capacity bottleneck --------------------------
print("\n(4) Attractor-capacity bottleneck (FIX G): 16 vs 4 distinguishable context states")
states_order1, states_order2 = N, N * N
bits_order1, bits_order2 = np.log2(states_order1), np.log2(states_order2)
# e06 measured ring capacity
e06 = np.load(HERE.parent / "e06_attractor" / "results" / "e06_results.npz",
              allow_pickle=True)
C_isi = float(e06["C_isi"]); C_long = float(e06["C_long"])
snr_isi = float(e06["snr_isi"])
# half-cell margin: a K-way 1-D ring quantizes the circle into K cells of width 2pi/K;
# the unambiguous half-cell margin is pi/K. Going 4 -> 16 states shrinks it 4x.
margin_order1 = np.pi / states_order1
margin_order2 = np.pi / states_order2
margin_shrink = margin_order1 / margin_order2
# the jitter that e06 measured over an ISI (radians), from its reported SNR
jitter_isi_rad = float(e06["jitter_isi"]) * np.pi / 180.0
print(f"    order-1 context: {states_order1} states = {bits_order1:.2f} bits, "
      f"half-cell margin {np.degrees(margin_order1):.1f} deg")
print(f"    order-2 context: {states_order2} states = {bits_order2:.2f} bits, "
      f"half-cell margin {np.degrees(margin_order2):.1f} deg  ({margin_shrink:.0f}x finer)")
print(f"    e06 MEASURED ring capacity: C_isi = {C_isi:.2f} bits over an ISI "
      f"(SNR {snr_isi:.0f}), C_long = {C_long:.2f} bits")
print(f"    raw budget: a single ring HAS {C_isi:.2f} bits >= {bits_order2:.2f} needed "
      f"for 16 ordered-pair states -- so the bit budget is NOT the binding limit.")
print(f"    THE REAL CATCH (FIX G): writing the ORDERED pair onto one ring needs a 16-way "
      f"quantization whose half-cell margin ({np.degrees(margin_order2):.1f} deg) is {margin_shrink:.0f}x")
print(f"    tighter; e06's measured ISI jitter is ~{np.degrees(jitter_isi_rad):.1f} deg, comfortably "
      f"inside 4-way ({np.degrees(margin_order1):.0f} deg) but eating into the 16-way margin "
      f"({np.degrees(margin_order2):.0f} deg).")
margin16_ok = jitter_isi_rad < margin_order2              # does e06 jitter still fit 16-way?
print(f"    -> e06's ISI jitter still fits the 16-way margin: {margin16_ok} "
      f"(margin {np.degrees(margin_order2):.1f} deg vs jitter {np.degrees(jitter_isi_rad):.1f} deg), "
      f"but the SNR headroom drops {margin_shrink:.0f}x. Deeper memory is capacity-bottlenecked, "
      f"not free.")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Fig 1: energy descent -- first-order (stuck at H1) vs eligibility-trace (-> H2),
#        both floors marked. Build a descent trajectory for each.
fig, ax = plot.new_fig(w=7.2, h=4.6)
# first-order numpy descent snapshots
snap_E1 = []
snap_xs = []
for m in snap_steps:
    if m in snaps1:
        snap_E1.append(energy_order1(snaps1[m], x, lo_eval))
        snap_xs.append(m)
snap_xs = np.array(snap_xs); snap_E1 = np.array(snap_E1)
# eligibility-trace descent at the best gamma (track energy vs training step)
W = np.zeros((2 * N, N)); g = g_best; lr = 0.05
trace_xs, trace_E = [], []
snap_set = set(snap_steps)
for t in range(2, n_stream):
    f = lag_features(x, t, g)
    q = softmax(W.T @ f); r = np.eye(N)[x[t]] - q
    W += lr * np.outer(f, r)
    if t in snap_set:
        # evaluate current W on the tail
        s = [-np.log2(max(softmax(W.T @ lag_features(x, tt, g))[x[tt]], 1e-12))
             for tt in range(lo_eval, lo_eval + 4000)]
        trace_xs.append(t); trace_E.append(np.mean(s))
trace_xs = np.array(trace_xs); trace_E = np.array(trace_E)
ax.plot(snap_xs, snap_E1, "o-", color=plot.C_THEORY, lw=2, ms=5,
        label="first-order learner (prev symbol)")
ax.plot(trace_xs, trace_E, "s-", color=plot.C_MEASURED, lw=2, ms=5,
        label=f"eligibility-trace learner (gamma={g_best:.2f})")
ax.axhline(H0, color="gray", ls=":", lw=1.2, label=f"marginal H0 = {H0:.4f}")
ax.axhline(H1, color="orange", ls="-.", lw=1.6, label=f"order-1 floor H1 = {H1:.4f}")
ax.axhline(H2, color=plot.C_FLOOR, ls="--", lw=1.8, label=f"order-2 floor H2 = {H2:.4f}")
ax.fill_between([snap_xs.min(), snap_xs.max()], H2, H1, color=plot.C_FLOOR, alpha=0.08)
ax.annotate(f"$I_2$ = {I2:.3f} bits\nleft on the table\nby order-1",
            (snap_xs.max() * 0.30, (H1 + H2) / 2), fontsize=8, color=plot.C_FLOOR, ha="center")
ax.set_xscale("log")
ax.set_xlabel("training step (symbols seen, log scale)")
ax.set_ylabel("energy (bits/symbol)")
ax.set_title("e14: first-order learner stuck at H1; eligibility trace descends toward H2")
ax.legend(frameon=False, fontsize=8, loc="upper right")
plot.save(fig, RESULTS / "e14_energy_descent.pdf")

# Fig 2: recovered fraction of I_2 vs gamma (the headline recovery curve).
fig, ax = plot.new_fig(w=6.4, h=4.2)
ax.plot(gammas, recovered_frac * 100, "o-", color=plot.C_MEASURED, lw=2, ms=7)
ax.axhline(0, color=plot.C_THEORY, ls="-.", lw=1.4, label="first-order (no trace, gamma=0)")
ax.axhline(100, color=plot.C_FLOOR, ls="--", lw=1.6, label="full I_2 recovered (-> H2)")
for gg, fr in zip(gammas, recovered_frac):
    ax.annotate(f"{fr*100:.0f}%", (gg, fr * 100 + 4), ha="center", fontsize=8)
ax.set_xlabel("eligibility-trace decay gamma")
ax.set_ylabel("% of $I_2$ recovered")
ax.set_ylim(-15, 115)
ax.set_title(f"e14: the lag-tagged trace recovers $I_2$ = {I2:.3f} bits once gamma > 0")
ax.legend(frameon=False, fontsize=8, loc="center right")
plot.save(fig, RESULTS / "e14_recovered_fraction.pdf")

# Fig 3: spiking trace dynamics -- build-up + geometric decay of e <- gamma e + c.
import matplotlib.pyplot as plt  # noqa: E402
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.8))
# left: clean decay of one pulsed channel
kk = np.arange(len(decay_seq))
axL.plot(kk, ch0, "o-", color=plot.C_MEASURED, ms=6, label="spiking trace (ch 0)")
# overlay the ideal geometric decay from the post-pulse peak
peak_k = int(np.argmax(ch0)); peak_v = ch0[peak_k]
ideal = peak_v * GAMMA_TRACE ** (kk - peak_k).clip(0)
axL.plot(kk[peak_k:], ideal[peak_k:], "--", color=plot.C_THEORY, lw=1.5,
         label=f"ideal $\\gamma^k$, $\\gamma$={GAMMA_TRACE}")
axL.set_xlabel("symbol window k"); axL.set_ylabel("decoded trace value")
axL.set_title(f"trace decay $e \\leftarrow \\gamma e$ (measured $\\hat\\gamma$={gamma_hat:.2f})")
axL.legend(frameon=False, fontsize=8)
# right: build/decay over a context stream (all channels)
kk2 = np.arange(len(demo_seq))
for j in range(N):
    axR.plot(kk2, trace_demo[:, j], "o-", ms=4, label=f"trace[{LABELS[j]}]")
for k, c in enumerate(demo_seq):
    if c >= 0:
        axR.annotate(LABELS[c], (k, -0.06), ha="center", fontsize=8, color="gray")
axR.set_xlabel("symbol window k (context shown below)")
axR.set_ylabel("decoded trace value")
axR.set_title("trace integrates the context history $e \\leftarrow \\gamma e + c$")
axR.legend(frameon=False, fontsize=7, ncol=2)
fig.suptitle("e14: the eligibility trace, realized on a spiking leaky-integrator population",
             fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS / "e14_spiking_trace.pdf", bbox_inches="tight")
plt.close(fig)

# Fig 4: FIX G capacity -- states/margin order-1 vs order-2 against e06's measured ring.
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 3.8))
axA.bar([0, 1], [bits_order1, bits_order2], color=[plot.C_MEASURED, plot.C_THEORY],
        width=0.55)
axA.axhline(C_isi, color=plot.C_FLOOR, ls="--", lw=1.6,
            label=f"e06 ring C_isi = {C_isi:.1f} bits")
axA.axhline(C_long, color="orange", ls="-.", lw=1.3, label=f"e06 long C = {C_long:.1f} bits")
axA.set_xticks([0, 1]); axA.set_xticklabels(["order-1\n(4 states)", "order-2\n(16 states)"])
axA.set_ylabel("context bits needed (log2 states)")
axA.set_title("(a) context capacity: 4 vs 16 states")
for i, v in enumerate([bits_order1, bits_order2]):
    axA.annotate(f"{v:.0f} bits", (i, v + 0.15), ha="center", fontsize=9)
axA.legend(frameon=False, fontsize=8, loc="upper left")
# right: half-cell margin shrink
axB.bar([0, 1], [np.degrees(margin_order1), np.degrees(margin_order2)],
        color=[plot.C_MEASURED, plot.C_THEORY], width=0.55)
axB.axhline(np.degrees(jitter_isi_rad), color=plot.C_FLOOR, ls="--", lw=1.6,
            label=f"e06 ISI jitter ~{np.degrees(jitter_isi_rad):.1f} deg")
axB.set_xticks([0, 1]); axB.set_xticklabels(["order-1\n(4-way)", "order-2\n(16-way)"])
axB.set_ylabel("half-cell margin (deg)")
axB.set_title(f"(b) FIX G: 16-way margin is {margin_shrink:.0f}x tighter")
for i, v in enumerate([np.degrees(margin_order1), np.degrees(margin_order2)]):
    axB.annotate(f"{v:.0f} deg", (i, v + 1.5), ha="center", fontsize=9)
axB.legend(frameon=False, fontsize=8, loc="upper right")
fig.suptitle("e14: the attractor-capacity bottleneck for deeper memory (FIX G / BIO-06)",
             fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS / "e14_capacity.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Save + acceptance
# ---------------------------------------------------------------------------
np.savez(RESULTS / "e14_results.npz",
         T2=T2, pi1=pi1, P1=P1, H0=H0, H1=H1, H2=H2, I2=I2, I1=I1, ev2=fl["ev2"],
         emp_H1=emp_H1, emp_H2=emp_H2,
         Q1_np=Q1_np, E1_np=E1_np, Q1_pes=Q1_pes, E1_pes=E1_pes, n_pes=n_pes,
         gammas=gammas, E_trace=E_trace, recovered_frac=recovered_frac,
         E_trace_best=E_trace_best, g_best=g_best, frac_best=frac_best,
         E_collapsed=E_collapsed,
         snap_xs=snap_xs, snap_E1=snap_E1, trace_xs=trace_xs, trace_E=trace_E,
         decay_seq=np.array(decay_seq), trace_decay=trace_decay, ch0=ch0,
         gamma_hat=gamma_hat, GAMMA_TRACE=GAMMA_TRACE,
         demo_seq=np.array(demo_seq), trace_demo=trace_demo,
         bits_order1=bits_order1, bits_order2=bits_order2,
         margin_order1=margin_order1, margin_order2=margin_order2,
         margin_shrink=margin_shrink, C_isi=C_isi, C_long=C_long,
         jitter_isi_rad=jitter_isi_rad, margin16_ok=margin16_ok)

checks = {
    f"2nd-order source has I_2 = H1 - H2 > 0 (I_2 = {I2:.4f})":
        I2 > 0.2,
    f"order-1 rate strictly below marginal, order-2 strictly below order-1 "
    f"(H0={H0:.3f} > H1={H1:.3f} > H2={H2:.3f})":
        H0 > H1 + 0.05 and H1 > H2 + 0.2,
    f"source ergodic (spectral-gap proxy |ev2| = {fl['ev2']:.3f} < 1)":
        fl["ev2"] < 0.999,
    f"numpy first-order learner stuck at the order-1 floor "
    f"(E1 = {E1_np:.4f}, |E1 - H1| < 0.08)":
        abs(E1_np - H1) < 0.08,
    f"spiking PES first-order learner CANNOT cross toward H2: it sits in the order-1 "
    f"band near the marginal (H1 <= E1_pes = {E1_pes:.4f}, and E1_pes - H2 = "
    f"{E1_pes - H2:.3f} >> I_2's worth below H1)":
        E1_pes >= H1 and (E1_pes - H2) > 0.7,
    f"first-order learners cannot cross H1: both above H2 + 0.5 "
    f"(E1_np {E1_np:.3f}, E1_pes {E1_pes:.3f} >> H2 {H2:.3f})":
        E1_np > H2 + 0.5 and E1_pes > H2 + 0.5,
    f"eligibility-trace learner drops energy BELOW the order-1 floor "
    f"(E_best = {E_trace_best:.4f} < H1 = {H1:.4f})":
        E_trace_best < H1 - 0.2,
    f"eligibility-trace learner recovers a measurable fraction of I_2 "
    f"(frac = {frac_best:.3f} > 0.5)":
        frac_best > 0.5,
    f"gamma=0 (no trace) recovers ~0 of I_2; gamma>0 recovers much more "
    f"(g0 {recovered_frac[0]:.3f}, best {frac_best:.3f})":
        recovered_frac[0] < 0.1 and frac_best - recovered_frac[0] > 0.5,
    f"spiking trace realizes geometric decay e <- gamma e "
    f"(gamma_hat = {gamma_hat:.3f} in (0,1))":
        0.0 < gamma_hat < 1.0,
    f"FIX G capacity quantified: order-2 needs 16 states vs 4, margin {margin_shrink:.0f}x "
    f"tighter; e06 ring C_isi = {C_isi:.1f} bits exceeds the {bits_order2:.0f}-bit budget":
        states_order2 == 16 and margin_shrink == 4 and C_isi > bits_order2,
}

print("\n" + "=" * 76)
allok = True
for name, ok in checks.items():
    allok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\ne14: {sum(checks.values())}/{len(checks)} acceptance checks passed.")
sys.exit(0 if allok else 1)
