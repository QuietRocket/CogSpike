#import "../report_style.typ": *
#show: setup

#report_header("e03", "The exact simplex normalizer, bought with neurons",
  "Tier 1 · predictor sub-circuit · paper claim #7 (softmax vs divisive normalization)")

#claim[
  The predicted distribution is $q_j = "softmax"_j (W c_t) = e^(a_j) slash sum_k
  e^(a_k)$, and the partition-of-unity invariant $sum_j q_j = 1$ holds *exactly*
  because of the softmax: it is an exact normalizer onto the probability simplex
  (paper Definition, eq. softmax). Carandini–Heeger divisive normalization
  $r_i = a_i slash (sigma + sum_j a_j)$ sums to $sum_i a_i slash (sigma + sum_i
  a_i) < 1$ for every $sigma > 0$, reaching $1$ only as $sigma arrow 0$; it is the
  *biophysical approximation* to softmax, a gain control, *not* an exact
  normalizer (paper Proposition). In `float64` the partition of unity is free. In
  the NEF the softmax must be *decoded from a finite, heterogeneous spiking
  population*, so $sum_j q_j = 1$ stops being an algebraic identity and becomes a
  *representational* quantity — true only up to the population's decode error.
]

= What we built

A single NEF ensemble (`build_softmax_predictor`) represents an $N = 4$ logit
vector $a = W c$ in a radius-$4$ ball and decodes $q = "softmax"(a)$ off the
spiking population. We sweep $n_("neurons") in {50, 100, 200, 400, 800, 1600}$,
and for each size run a fixed test set of $12$ logit vectors covering the
representable ball, hold each constant for $0.3$ s, and average the decoded $q$
over the settled tail (rejecting the synaptic transient). Against the analytic
`spikecoder.information.softmax` (the `float64` validator) we measure (a) the
per-component RMSE of the decoded $q$ and (b) the normalization defect $|sum_j q_j
- 1|$. Separately we build the divisive-normalization node (`build_divisive_norm`)
on a fixed nonnegative drive and sweep $sigma$. All code is `e03_softmax/run.py`.

= The partition of unity is now representational, not exact

#finding[
  The NEF decodes the softmax with an error that falls as the textbook
  $1 slash sqrt(n_("neurons"))$ rate. The per-component RMSE drops from $0.056$ at
  $50$ neurons to $0.0095$ at $1600$, and is *$0.0229$ at $n = 400$* — already
  under the $0.05$ target — with a log-log slope of *$-0.522$* (the NEF ideal is
  $-1 slash 2$). The normalization defect $|sum_j q_j - 1|$ falls from $0.018$ to
  $0.0011$ over the same sweep.

  #table(columns: 5, align: (center,)*5, stroke: 0.5pt + luma(200),
    table.header[$n_("neurons")$][RMSE$(q)$][$|sum q - 1|$][$-log_2|sum q - 1|$ (bits)][bits/neuron],
    [50],   [0.0560], [0.0176], [5.83], [$1.17 times 10^(-1)$],
    [100],  [0.0440], [0.0093], [6.74], [$6.74 times 10^(-2)$],
    [200],  [0.0325], [0.0070], [7.15], [$3.57 times 10^(-2)$],
    [400],  [0.0229], [0.0032], [8.31], [$2.08 times 10^(-2)$],
    [800],  [0.0139], [0.0030], [8.38], [$1.05 times 10^(-2)$],
    [1600], [0.0095], [0.0011], [9.83], [$6.15 times 10^(-3)$],
  )
]

#figure(image("results/e03_nef_scaling.pdf", width: 80%),
  caption: [NEF softmax error vs ensemble size, log-log. Per-component RMSE (blue)
    and normalization defect $|sum q - 1|$ (green) both decrease with $n_("neurons")$;
    the RMSE tracks the $1 slash sqrt(N)$ guide (red dashed) with fitted slope
    $-0.522$, and is below the $0.05$ target (grey dotted) for $n >= 400$.])

#gap[
  This is the headline spiking-reality gap of the claim. The numpy validator
  computes $sum_j q_j = 1$ to machine precision *for free* — it is an algebraic
  identity of the softmax. On the spiking substrate that identity is *bought with
  neurons*: the partition of unity holds only to the population's decode accuracy,
  and buying it tighter costs $tilde.op 1 slash sqrt(N)$ more units. Read as a *precision*,
  the normalization defect is $-log_2 |sum q - 1|$ bits of partition-of-unity
  accuracy: $5.83$ bits at $50$ neurons rising to $9.83$ bits at $1600$. At the
  measured rate, one further bit of normalization precision costs roughly an
  order of magnitude more neurons. *The exact simplex invariant the paper leans on
  is, in the substrate, an asymptotic property of the representation, not a free
  identity* — but it is a benign one: it converges, monotonically, at the NEF's
  guaranteed rate.
]

#honest[
  The normalization-defect curve falls *faster* than $1 slash sqrt(N)$ — its
  log-log slope is $-0.745$, steeper than the $-0.522$ of the per-component RMSE.
  This is expected and not a tighter law: $sum_j q_j - 1 = sum_j (q_j^("dec") -
  q_j^("true"))$ is a *signed sum* of the four per-component errors, so the partly
  independent decode errors cancel rather than add. The defect is therefore a
  partially-cancelled residual that shrinks faster than the (always-positive)
  RMSE; the honest single rate for "how well does the population represent the
  softmax" is the RMSE slope $-0.522$, which is the one we test against the NEF's
  $-1 slash 2$ prediction. The defect's faster decay is a bonus, not a separate
  scaling law. (The two slopes are computed over only six grid points, so treat
  the second decimal as indicative.)
]

= Divisive normalization sums to strictly less than one

#finding[
  On the representative drive $a = [2, 1, 0.5, 0.5]$ (total $sum a = 4$), the
  divisive-normalization output sums to $sum_i r_i = sum a slash (sigma + sum a)$,
  *strictly below $1$ for every $sigma > 0$*: $0.800$ at $sigma = 1$, $0.976$ at
  $sigma = 0.1$, rising to $0.9988$ at $sigma = 0.005$. The measured sum matches
  the analytic $sum a slash (sigma + sum a)$ to $< 10^(-3)$ at every $sigma$ — the
  divisive-norm node is a *deterministic* gain computation, so (unlike the spiking
  softmax decode) it carries no representational error of its own. This is the
  paper's Proposition realized: divisive normalization is *not*
  a partition of unity; it approaches one only in the $sigma arrow 0$ limit.
]

#figure(image("results/e03_divnorm_sigma.pdf", width: 74%),
  caption: [Divisive-normalization sum $sum_i r_i$ vs the semi-saturation constant
    $sigma$. Measured (blue) lands exactly on the analytic $sum a slash (sigma +
    sum a)$ (red); it is $< 1$ for all $sigma > 0$ and rises toward the softmax's
    exact $sum q = 1$ (green dashed) only as $sigma arrow 0$. The gap $sigma
    slash (sigma + sum a)$ is the unspent normalization budget.])

#intuition[
  $sigma$ is a fixed leak added to the normalizing pool. Softmax normalizes by the
  *exact* pool $sum_k e^(a_k)$ with no additive slack, so it lands on the simplex;
  divisive normalization normalizes by $sigma$ plus the linearized pool, so it
  always leaves a fraction $sigma slash (sigma + sum_j a_j)$ unspent. The two agree
  in the high-drive / small-$sigma$ regime, which is exactly why the paper can call
  divisive normalization the $sigma arrow 0$ *realization* of the softmax while
  keeping the partition-of-unity obligation on the softmax itself.
]

= Contrast on one vector

#figure(image("results/e03_contrast.pdf", width: 80%),
  caption: [Same vector, three normalizers. Analytic softmax (dark blue) sums to
    $1.000000$ exactly; the NEF-decoded softmax at $n = 800$ (light blue) sums to
    $1.000058$ — exact up to the representation; divisive normalization at
    $sigma = 0.1$ (red) sums to $0.9756 < 1$, leaving a visible deficit on every
    component.])

#finding[
  For $a = [2, 1, 0.5, 0.5]$: analytic softmax sums to $1.000000$ (exact simplex),
  the $n = 800$ NEF softmax sums to $1.000058$ (representational, off by
  $5.8 times 10^(-5)$), and divisive normalization at $sigma = 0.1$ sums to
  $0.9756$. The two maps are genuinely different objects — softmax is the exact
  exponential-family normalizer whose logits read as log-odds (used by the
  calibration drive of e02), divisive normalization is a gain control that only
  approximates it — exactly the separation the paper's fix A1 insists on.
]

= Acceptance

#accept_table((
  (true, [NEF softmax RMSE decreases monotonically with $n_("neurons")$ ($0.056 arrow 0.0095$)]),
  (true, [NEF softmax RMSE $< 0.05$ at $n_("neurons") >= 400$ ($0.0229$ at $400$)]),
  (true, [RMSE log-log slope $approx -1 slash 2$ (measured $-0.522$, NEF scaling)]),
  (true, [normalization defect $|sum q - 1|$ shrinks with $n_("neurons")$ ($0.018 arrow 0.0011$)]),
  (true, [divisive norm sums to $< 1$ for every $sigma > 0$ (the paper's Proposition)]),
  (true, [divisive norm sum $arrow 1$ as $sigma arrow 0$ ($0.800 arrow 0.9988$)]),
  (true, [softmax (numpy) sums to $1$ exactly — the exact simplex normalizer]),
  (true, [divisive-norm node matches its analytic sum to $< 10^(-3)$ (deterministic gain)]),
))

#finding[
  *8/8 passed.* The paper's gain-control distinction holds in the substrate and is
  sharpened by it. Softmax is the exact simplex normalizer ($sum q = 1$ to machine
  precision); divisive normalization is its $sigma arrow 0$ approximation
  ($sum r < 1$, deficit $sigma slash (sigma + sum a)$). The one genuinely new
  spiking fact: in the NEF the softmax's partition of unity is no longer a free
  algebraic identity but a *representational* quantity, bought with neurons at the
  $1 slash sqrt(N)$ rate — $0.0229$ RMSE and $5.83$–$9.83$ bits of normalization
  precision over $50$–$1600$ units. This is the most reliable rung in the suite:
  the only "error" is the price of representing an exact identity in finite spikes,
  and it converges cleanly.
]
