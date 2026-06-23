//! A 1-D synthetic event camera (DVS) with a predictive-inhibitory compressor, ported
//! from `research/compression/nengo_experiments/e16_event_stream/run.py`.
//!
//! A bright bar translates across a pixel strip. A pixel emits an **ON** event on a
//! rising brightness edge (the bar arrives) and an **OFF** event on a falling edge (it
//! leaves) -- so steady motion emits only the two moving edges: a sparse, predictable
//! stream. A recurrent shift-predictor anticipates each event from the previous frame's
//! events and the known velocity, and *inhibits* the predicted pixels. Only the
//! **unpredicted** events survive: the residual IS the compressed stream.
//!
//! In e16 the inhibitory subtraction is a Nengo LIF soma bank; because the event drive
//! `J_ON = 6` is far above threshold `theta = 1` with a per-frame reset, the soma fires
//! exactly when `raw - predicted > 0`. So the residual is the pure integer identity
//! `resid = raw AND NOT predicted`, which this module computes directly (WASM-safe,
//! byte-exact to the e16 oracle). e16 separately validated (9/9) that the spiking soma
//! bank realizes this identity.

use std::collections::VecDeque;

use rand::SeedableRng as _;
use rand::rngs::StdRng;

/// Pixels in the 1-D array (e16 default).
pub const P_PIXELS: usize = 24;
/// Bar width in pixels (e16 default).
pub const BAR_W: usize = 4;
/// Programmed steady bar velocity (pixels / frame).
pub const V: i64 = 1;

/// A one-frame event map for a single polarity (one `u8` in `{0,1}` per pixel).
pub type Row = Vec<u8>;
/// A movie of per-frame event maps (`n_frames` rows of `P` pixels).
pub type Movie = Vec<Row>;

/// Brightness of a width-`w` bar with left edge at `pos` (wrapping); `pos < 0` = absent.
#[must_use]
pub fn bar_brightness_frame(pos: i64, p: usize, w: usize) -> Row {
    let mut b = vec![0u8; p];
    if pos >= 0 && p > 0 {
        let base = pos.rem_euclid(p as i64) as usize;
        for k in 0..w {
            if let Some(cell) = b.get_mut((base + k) % p) {
                *cell = 1;
            }
        }
    }
    b
}

/// Brightness movie for a position schedule.
#[must_use]
pub fn brightness(positions: &[i64], p: usize, w: usize) -> Movie {
    positions
        .iter()
        .map(|&pos| bar_brightness_frame(pos, p, w))
        .collect()
}

/// Per-frame ON/OFF event maps from a brightness movie. A pixel emits ON when its
/// brightness rises `0 -> 1` and OFF when it falls `1 -> 0`; frame 0 is compared to dark.
#[must_use]
pub fn events_from_brightness(b: &[Row]) -> (Movie, Movie) {
    let p = b.first().map_or(0, Vec::len);
    let mut prev = vec![0u8; p];
    let mut on = Vec::with_capacity(b.len());
    let mut off = Vec::with_capacity(b.len());
    for frame in b {
        on.push(
            frame
                .iter()
                .zip(&prev)
                .map(|(&c, &pv)| u8::from(c > pv))
                .collect(),
        );
        off.push(
            frame
                .iter()
                .zip(&prev)
                .map(|(&c, &pv)| u8::from(c < pv))
                .collect(),
        );
        prev = frame.clone();
    }
    (on, off)
}

/// `np.roll` semantics: `rolled[i] = a[(i - k) mod n]` (shift right by `k`, wrapping).
#[must_use]
pub fn roll(a: &[u8], k: i64) -> Row {
    let n = a.len() as i64;
    if n == 0 {
        return Row::new();
    }
    (0..a.len())
        .map(|i| {
            let src = ((i as i64 - k).rem_euclid(n)) as usize;
            a.get(src).copied().unwrap_or(0)
        })
        .collect()
}

/// Predict frame `k`'s events by rolling frame `k-1`'s events by the per-frame velocity
/// `v_hat[k]`. Frame 0 predicts nothing.
#[must_use]
pub fn predict_events(on: &[Row], off: &[Row], v_hat: &[i64]) -> (Movie, Movie) {
    let n = on.len();
    let p = on.first().map_or(0, Vec::len);
    let predict = |src: &[Row]| -> Movie {
        (0..n)
            .map(|k| match (k.checked_sub(1), v_hat.get(k)) {
                (Some(prev), Some(&vk)) => roll(src.get(prev).map_or(&[][..], Vec::as_slice), vk),
                _ => vec![0u8; p],
            })
            .collect()
    };
    (predict(on), predict(off))
}

/// The residual identity: a pixel survives iff it had a raw event the predictor did not
/// anticipate -- `resid = raw AND NOT predicted`.
#[must_use]
pub fn residual_map(raw: &[Row], pred: &[Row]) -> Movie {
    raw.iter()
        .zip(pred)
        .map(|(rr, pr)| {
            rr.iter()
                .zip(pr)
                .map(|(&r, &p)| u8::from(r == 1 && p == 0))
                .collect()
        })
        .collect()
}

/// Total set bits across a movie (event count).
#[must_use]
pub fn total(m: &[Row]) -> u64 {
    m.iter().flatten().map(|&v| u64::from(v)).sum()
}

/// Count of pixels where both maps are set (predictor hits).
#[must_use]
pub fn overlap(a: &[Row], b: &[Row]) -> u64 {
    a.iter()
        .zip(b)
        .flat_map(|(ar, br)| {
            ar.iter()
                .zip(br)
                .map(|(&x, &y)| u64::from(x == 1 && y == 1))
        })
        .sum()
}

/// A novelty marker in the canonical schedule (label, frame index).
pub type Novelty = (&'static str, usize);

/// The canonical e16 schedule: 3 dark frames, an onset, +V steady motion, a reversal to
/// -V, then a velocity jump to +2V. Returns `(positions, novelty markers, true velocity)`.
#[must_use]
pub fn build_schedule() -> (Vec<i64>, Vec<Novelty>, Vec<i64>) {
    let p = P_PIXELS as i64;
    let mut pos: Vec<i64> = Vec::new();
    let mut vel: Vec<i64> = Vec::new();
    let mut novel: Vec<Novelty> = Vec::new();

    for _ in 0..3 {
        pos.push(-1);
        vel.push(0);
    }
    novel.push(("onset", pos.len()));
    pos.push(0);
    vel.push(0);
    let mut cur = 0_i64;
    for _ in 0..15 {
        cur = (cur + V).rem_euclid(p);
        pos.push(cur);
        vel.push(V);
    }
    novel.push(("reversal", pos.len()));
    for _ in 0..12 {
        cur = (cur - V).rem_euclid(p);
        pos.push(cur);
        vel.push(-V);
    }
    novel.push(("jump", pos.len()));
    for _ in 0..10 {
        cur = (cur + 2 * V).rem_euclid(p);
        pos.push(cur);
        vel.push(2 * V);
    }
    (pos, novel, vel)
}

/// Causal one-frame-lag velocity estimate `v_hat[k] = vel[k-1]` (the best a one-frame
/// memory can do; wrong for exactly one frame at each velocity change).
#[must_use]
pub fn estimate_velocity_lagging(vel: &[i64]) -> Vec<i64> {
    let mut v_hat = vec![0_i64; vel.len()];
    for (dst, w) in v_hat.iter_mut().skip(1).zip(vel.iter()) {
        *dst = *w;
    }
    v_hat
}

/// Summary of a predictive-compression run over a schedule.
#[derive(Clone, Debug)]
pub struct CompressionStats {
    /// Raw ON+OFF event count.
    pub raw_total: u64,
    /// Residual ON+OFF spike count after predictive subtraction.
    pub resid_total: u64,
    /// `raw_total / max(resid_total, 1)`.
    pub ratio: f64,
    /// Fraction of raw events the predictor anticipated.
    pub predictability: f64,
}

/// Run the full predictive compressor over a position schedule with a per-frame velocity
/// estimate, returning the residual maps and the compression statistics.
#[must_use]
pub fn compress(
    positions: &[i64],
    v_hat: &[i64],
    p: usize,
    w: usize,
) -> (Movie, Movie, CompressionStats) {
    let b = brightness(positions, p, w);
    let (on_raw, off_raw) = events_from_brightness(&b);
    let (on_pred, off_pred) = predict_events(&on_raw, &off_raw, v_hat);
    let on_resid = residual_map(&on_raw, &on_pred);
    let off_resid = residual_map(&off_raw, &off_pred);
    let raw_total = total(&on_raw) + total(&off_raw);
    let resid_total = total(&on_resid) + total(&off_resid);
    let hits = overlap(&on_raw, &on_pred) + overlap(&off_raw, &off_pred);
    let stats = CompressionStats {
        raw_total,
        resid_total,
        ratio: raw_total as f64 / resid_total.max(1) as f64,
        predictability: hits as f64 / raw_total.max(1) as f64,
    };
    (on_resid, off_resid, stats)
}

// ---------------------------------------------------------------------------
// Live, frame-stepped scene for the interactive playground.
// ---------------------------------------------------------------------------

/// One rendered frame of the live event camera (raw + residual, both polarities).
#[derive(Clone, Debug)]
pub struct DvsFrame {
    /// Raw ON events this frame.
    pub on_raw: Row,
    /// Raw OFF events this frame.
    pub off_raw: Row,
    /// Surviving ON residual after predictive subtraction.
    pub on_resid: Row,
    /// Surviving OFF residual.
    pub off_resid: Row,
    /// Whether this frame was hallucinated in open-loop dream mode (raw = the model's
    /// own prediction, residual identically zero).
    pub is_dream: bool,
}

/// An unbounded, frame-stepped event camera for the interactive view: a bar moves at a
/// tunable velocity with optional jitter, and a one-frame-lag predictor compresses it.
#[derive(Clone, Debug)]
pub struct DvsScene {
    /// Pixel count.
    pub p: usize,
    /// Bar width.
    pub w: usize,
    /// Base bar velocity (pixels / frame); user-tunable, may be negative.
    pub velocity: i64,
    /// Per-frame velocity-jitter probability of a random +/-1 hop (corrupts predictability).
    pub jitter: f64,
    /// Whether the predictive-inhibitory compressor is engaged.
    pub predictor_on: bool,
    /// Recent frames kept for the raster display (most recent at the back).
    pub frames: VecDeque<DvsFrame>,
    /// Cumulative raw event count (since reset).
    pub raw_total: u64,
    /// Cumulative residual spike count (since reset).
    pub resid_total: u64,
    pos: i64,
    prev_vel: i64,
    prev_bright: Row,
    prev_on: Row,
    prev_off: Row,
    rng: StdRng,
    // --- open-loop dream bookkeeping (inert unless dream_step() is driven) ---
    in_dream: bool,
    /// Velocity the predictor believed at dream onset; the dream glides at this.
    dream_vel: i64,
    /// Where the dreamer thinks the bar is.
    dreamed_pos: i64,
    prev_dream_bright: Row,
    frames_dreamt: u64,
    /// Circular pixel distance between the dreamed bar and the (unseen) real one.
    divergence: f64,
}

/// Frames retained in the live raster window.
pub const DVS_WINDOW: usize = 90;

impl DvsScene {
    /// A fresh scene with the e16 pixel geometry, moving right at +V, no jitter.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let p = P_PIXELS;
        Self {
            p,
            w: BAR_W,
            velocity: V,
            jitter: 0.0,
            predictor_on: true,
            frames: VecDeque::with_capacity(DVS_WINDOW),
            raw_total: 0,
            resid_total: 0,
            pos: 0,
            prev_vel: 0,
            prev_bright: vec![0u8; p],
            prev_on: vec![0u8; p],
            prev_off: vec![0u8; p],
            rng: StdRng::seed_from_u64(seed),
            in_dream: false,
            dream_vel: 0,
            dreamed_pos: 0,
            prev_dream_bright: vec![0u8; p],
            frames_dreamt: 0,
            divergence: 0.0,
        }
    }

    /// Live compression ratio (raw / residual) since reset.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        self.raw_total as f64 / self.resid_total.max(1) as f64
    }

    /// The current brightness frame (where the bar is right now).
    #[must_use]
    pub fn current_brightness(&self) -> &[u8] {
        &self.prev_bright
    }

    /// Advance one event-camera frame: move the bar, emit events, predictively subtract.
    pub fn step(&mut self) {
        use rand::Rng as _;
        // actual motion this frame = base velocity + optional jitter hop
        let mut step_vel = self.velocity;
        if self.jitter > 0.0 && self.rng.r#gen::<f64>() < self.jitter {
            step_vel += if self.rng.r#gen::<bool>() { 1 } else { -1 };
        }
        self.pos = (self.pos + step_vel).rem_euclid(self.p as i64);

        let bright = bar_brightness_frame(self.pos, self.p, self.w);
        let on_raw: Row = bright
            .iter()
            .zip(&self.prev_bright)
            .map(|(&c, &pv)| u8::from(c > pv))
            .collect();
        let off_raw: Row = bright
            .iter()
            .zip(&self.prev_bright)
            .map(|(&c, &pv)| u8::from(c < pv))
            .collect();

        // one-frame-lag predictor: roll the previous frame's events by last frame's velocity
        let (on_pred, off_pred) = if self.predictor_on {
            (
                roll(&self.prev_on, self.prev_vel),
                roll(&self.prev_off, self.prev_vel),
            )
        } else {
            (vec![0u8; self.p], vec![0u8; self.p])
        };
        let on_resid: Row = on_raw
            .iter()
            .zip(&on_pred)
            .map(|(&r, &p)| u8::from(r == 1 && p == 0))
            .collect();
        let off_resid: Row = off_raw
            .iter()
            .zip(&off_pred)
            .map(|(&r, &p)| u8::from(r == 1 && p == 0))
            .collect();

        self.raw_total +=
            total(std::slice::from_ref(&on_raw)) + total(std::slice::from_ref(&off_raw));
        self.resid_total +=
            total(std::slice::from_ref(&on_resid)) + total(std::slice::from_ref(&off_resid));

        self.prev_bright = bright;
        self.prev_on = on_raw.clone();
        self.prev_off = off_raw.clone();
        self.prev_vel = step_vel;

        self.frames.push_back(DvsFrame {
            on_raw,
            off_raw,
            on_resid,
            off_resid,
            is_dream: false,
        });
        if self.frames.len() > DVS_WINDOW {
            self.frames.pop_front();
        }
    }

    /// Where the dreamer currently believes the bar is (its hallucinated brightness).
    #[must_use]
    pub fn dreamed_brightness(&self) -> &[u8] {
        &self.prev_dream_bright
    }

    /// Circular pixel distance between the dreamed bar and the (unseen) real one.
    #[must_use]
    pub fn divergence(&self) -> f64 {
        self.divergence
    }

    /// Frames elapsed since the current dream began.
    #[must_use]
    pub fn frames_dreamt(&self) -> u64 {
        self.frames_dreamt
    }

    /// Whether the scene is currently dreaming (open-loop, sensory input cut).
    #[must_use]
    pub fn in_dream(&self) -> bool {
        self.in_dream
    }

    /// Enter open-loop dreaming: freeze the believed velocity and seed the dreamed bar at
    /// the agent's current best estimate of the bar's position.
    pub fn begin_dream(&mut self) {
        self.in_dream = true;
        self.dream_vel = self.prev_vel; // the velocity the lag-predictor actually uses
        self.dreamed_pos = self.pos;
        self.prev_dream_bright = self.prev_bright.clone();
        self.frames_dreamt = 0;
        self.divergence = 0.0;
    }

    /// Leave dreaming and resume observing the real world.
    pub fn wake(&mut self) {
        self.in_dream = false;
    }

    /// Advance one DREAM frame. The real bar keeps moving (ground truth the agent can no
    /// longer see); the dreamed bar glides at the frozen believed velocity. The emitted
    /// frame IS the dreamer's own prediction, so the residual is identically zero
    /// ("nothing surprises a dreamer"). The circular distance between the dreamed and
    /// real bar is accumulated for the overlay.
    pub fn dream_step(&mut self) {
        use rand::Rng as _;

        // (1) the REAL world advances exactly as in step() (jitter + base velocity), but
        //     the dreamer no longer observes it.
        let mut real_vel = self.velocity;
        if self.jitter > 0.0 && self.rng.r#gen::<f64>() < self.jitter {
            real_vel += if self.rng.r#gen::<bool>() { 1 } else { -1 };
        }
        self.pos = (self.pos + real_vel).rem_euclid(self.p as i64);
        let real_bright = bar_brightness_frame(self.pos, self.p, self.w);

        // (2) the DREAMED world: pure extrapolation at the frozen belief, no observation.
        self.dreamed_pos = (self.dreamed_pos + self.dream_vel).rem_euclid(self.p as i64);
        let dream_bright = bar_brightness_frame(self.dreamed_pos, self.p, self.w);

        // (3) emit the dreamed events (diff of the dreamed brightness) as this frame's raw.
        let on_raw: Row = dream_bright
            .iter()
            .zip(&self.prev_dream_bright)
            .map(|(&c, &pv)| u8::from(c > pv))
            .collect();
        let off_raw: Row = dream_bright
            .iter()
            .zip(&self.prev_dream_bright)
            .map(|(&c, &pv)| u8::from(c < pv))
            .collect();

        // (4) residual is identically zero: the predictor predicts its own dream perfectly.
        let on_resid = vec![0u8; self.p];
        let off_resid = vec![0u8; self.p];

        // (5) raw still accrues; residual does not -> the ratio inflates, a tell that the
        //     compressor has stopped learning (it only sees its own predictions).
        self.raw_total +=
            total(std::slice::from_ref(&on_raw)) + total(std::slice::from_ref(&off_raw));

        // (6) divergence: minimal circular distance between the dreamed and real positions.
        let p = self.p as i64;
        let gap = (self.dreamed_pos - self.pos).rem_euclid(p);
        self.divergence = gap.min(p - gap) as f64;

        // (7) roll both worlds forward; keep prev_* coherent so waking is clean (the first
        //     real step() after waking legitimately shows a big surprise burst).
        self.prev_dream_bright = dream_bright;
        self.prev_bright = real_bright;
        self.prev_on = on_raw.clone();
        self.prev_off = off_raw.clone();
        self.prev_vel = self.dream_vel;
        self.frames_dreamt += 1;

        self.frames.push_back(DvsFrame {
            on_raw,
            off_raw,
            on_resid,
            off_resid,
            is_dream: true,
        });
        if self.frames.len() > DVS_WINDOW {
            self.frames.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_schedule_compresses_and_concentrates() {
        let (positions, novel, vel) = build_schedule();
        assert_eq!(positions.len(), 41, "3 dark + 1 onset + 15 + 12 + 10");
        let v_hat = estimate_velocity_lagging(&vel);
        let (on_resid, off_resid, stats) = compress(&positions, &v_hat, P_PIXELS, BAR_W);

        // the predictive circuit compresses the redundant steady motion
        assert!(stats.ratio > 2.0, "compression ratio {} > 2", stats.ratio);
        assert!(stats.predictability > 0.5, "stream is redundant");

        // residual concentrates in the novelty windows (each marker + the next frame)
        let mut novel_mask = vec![false; positions.len()];
        for &(_, k) in &novel {
            for f in [k, k + 1] {
                if let Some(slot) = novel_mask.get_mut(f) {
                    *slot = true;
                }
            }
        }
        let resid_per_frame: Vec<u64> = on_resid
            .iter()
            .zip(&off_resid)
            .map(|(o, f)| total(std::slice::from_ref(o)) + total(std::slice::from_ref(f)))
            .collect();
        let in_novel: u64 = resid_per_frame
            .iter()
            .zip(&novel_mask)
            .filter(|&(_, &m)| m)
            .map(|(&r, _)| r)
            .sum();
        let frac = in_novel as f64 / stats.resid_total.max(1) as f64;
        assert!(frac >= 0.5, "residual concentrates at novelty: {frac}");
    }

    #[test]
    fn steady_motion_is_fully_predicted() {
        // a pure +V bar over many frames: after the first, the lagging predictor nails it
        let p = P_PIXELS as i64;
        let positions: Vec<i64> = (0..30).map(|k| (k * V).rem_euclid(p)).collect();
        let v_hat = vec![V; positions.len()];
        let (_, _, stats) = compress(&positions, &v_hat, P_PIXELS, BAR_W);
        // raw events keep coming but residual is near-zero once locked
        assert!(
            stats.ratio > 5.0,
            "steady motion compresses hard: {}",
            stats.ratio
        );
    }

    #[test]
    fn jitter_destroys_compression() {
        // more jitter -> the fixed predictor is wrong more often -> ratio falls toward 1
        let mut steady = DvsScene::new(7);
        steady.jitter = 0.0;
        let mut jittery = DvsScene::new(7);
        jittery.jitter = 1.0;
        for _ in 0..400 {
            steady.step();
            jittery.step();
        }
        assert!(
            steady.ratio() > jittery.ratio(),
            "steady {} should beat jittery {}",
            steady.ratio(),
            jittery.ratio()
        );
        assert!(steady.ratio() > 2.0, "steady motion compresses");
    }

    #[test]
    fn roll_matches_numpy_semantics() {
        // np.roll([1,0,0,0], 1) == [0,1,0,0]
        assert_eq!(roll(&[1, 0, 0, 0], 1), vec![0, 1, 0, 0]);
        assert_eq!(roll(&[1, 0, 0, 0], -1), vec![0, 0, 0, 1]);
        assert_eq!(roll(&[1, 2, 3, 4], 2), vec![3, 4, 1, 2]);
    }

    #[test]
    fn dream_has_zero_residual_and_peels_away_on_reversal() {
        let mut s = DvsScene::new(7);
        s.velocity = 2;
        s.jitter = 0.0;
        for _ in 0..10 {
            s.step();
        }
        assert_eq!(s.prev_vel, 2, "predictor believes +2 px/frame");

        s.begin_dream();
        s.velocity = -2; // the unseen world reverses while the agent dreams
        let resid_before = s.resid_total;
        let mut max_div = 0.0_f64;
        for k in 1..=20 {
            s.dream_step();
            let f = s.frames.back().expect("a dream frame");
            assert!(f.is_dream, "dream frame is flagged");
            assert!(
                f.on_resid.iter().all(|&b| b == 0) && f.off_resid.iter().all(|&b| b == 0),
                "nothing surprises a dreamer: residual is identically zero"
            );
            assert_eq!(s.frames_dreamt(), k);
            max_div = max_div.max(s.divergence());
        }
        assert_eq!(
            s.resid_total, resid_before,
            "no learning signal accrues in a dream"
        );
        // dream goes +2, reality goes -2 => 4 px/frame apart; on a 24-ring the circular
        // distance climbs to near its maximum (12) before wrapping.
        assert!(
            max_div >= 8.0,
            "dream provably peeled away from reality (max divergence {max_div})"
        );
    }

    #[test]
    fn dream_glides_at_frozen_belief_not_the_live_slider() {
        let mut s = DvsScene::new(7);
        s.velocity = 1;
        s.jitter = 0.0;
        for _ in 0..5 {
            s.step();
        }
        s.begin_dream();
        assert_eq!(s.dream_vel, 1, "belief frozen at +1 at onset");
        let before = s.dreamed_pos;
        s.velocity = -3; // change the (unseen) world AFTER the dream began
        s.dream_step();
        assert_eq!(
            (s.dreamed_pos - before).rem_euclid(s.p as i64),
            1,
            "the dream still glides at its frozen +1 belief, ignoring the -3 slider"
        );
    }
}
