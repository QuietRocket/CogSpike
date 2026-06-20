# e16 — A neuromorphic event stream: spikes-as-bits leaves the toy chain

**Tier 4 capstone (the realistic anchor).** The momentum rover (e01–e12) is a rehearsal;
the paper's real target is data that *is already spikes* — a dynamic-vision-sensor /
event-camera stream whose pixels emit events only on local brightness *change*. We build
a tiny synthetic 1-D event camera (`P=24` pixels, a width-4 bar translating, wrapping,
reversing, and jumping) that emits ON/OFF events on rising/falling edges — a sparse
(`5%`), spatiotemporally redundant (`0.895` predictable) spike stream. A recurrent
predictive circuit applies the **e07 inhibitory-subtraction sign** on the spatial event
field: a shift predictor (adaptive one-frame-lag velocity model) pre-charges the expected
next edge motion with inhibitory feedback, so each LIF soma integrates only the
**residual** `J_eff = J·(raw − predicted)` — correctly-anticipated events meet balanced
E/I and stay silent, unpredicted ones fire. Result: **98 raw events → 12 residual spikes
(8.17× compression)**, with **100% of the residual on the three unpredictable transitions**
(onset, reversal, velocity jump) and steady motion cancelled to zero (80×). A velocity-
jitter sweep shows the compression ratio falling from **16.4× to 1.31×** as predictability
decays from 0.94 to 0.24 (`r = +0.84`): compression cashes out exactly the redundancy
present. Honestly framed as a low-dimensional rehearsal — the predictor's velocity is
provided not learned, and a static-`+V` contrast (1.92×) shows the compression is the
*motion model's* doing. **9/9 acceptance checks pass.**

Run: `uv run python run.py` → `results/` + three PDFs;
`typst compile --root .. report.typ report.pdf`.
