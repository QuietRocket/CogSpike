//! The spike-latency view: first-spike latency = surprisal, made literal.
//!
//! For the live prediction `q`, each of the four symbol-neurons gets the calibration
//! drive `J = theta/(1 - q^alpha)` and its membrane ramps from rest; the neuron with the
//! highest `q` (the most expected symbol) has the strongest drive, so it crosses threshold
//! FIRST. The wait until that first spike is exactly the Shannon code length
//! `t*(q) = -lambda log2 q`. Early in learning all four ramps are bunched (a near-tie =
//! no compression); as the delta rule sharpens `q`, the expected symbol pulls ahead.

use std::collections::VecDeque;
use std::time::Duration;

use cog_spike::coder::{
    LABELS, LAMBDA, Q_CLIP_HI, Q_CLIP_LO, RHEOBASE_CEILING, TAU_RC, THETA, calibration_drive,
    nengo_first_spike_time, one_hot,
};
use cog_spike::gym::scenario::{self, Scenario};
use cog_spike::gym::{Agent as _, DeltaAgent, OnlineRover};
use egui::{Color32, RichText};
use egui_plot::{HLine, Legend, Line, Plot, PlotPoints, Points, VLine};

use crate::app::TemplateApp;

const WINDOW: usize = 2000;
const GREEN: Color32 = Color32::from_rgb(90, 210, 140);
const RED: Color32 = Color32::from_rgb(235, 120, 110);
/// Per-neuron ramp colours (U/D/L/R).
const NEURON_COLORS: [Color32; 4] = [
    Color32::from_rgb(90, 170, 255),
    Color32::from_rgb(255, 170, 90),
    Color32::from_rgb(120, 210, 130),
    Color32::from_rgb(220, 130, 220),
];

/// Upper cap (wall-clock seconds) on the slow-motion replay of one symbol's spike race; the
/// actual replay is at most half the per-symbol hold so the reveal is always shown afterwards.
const RACE_SECS_CAP: f64 = 1.1;

/// Lower bound (model-seconds) on the displayed spike-time span, so an all-fast symbol still
/// spreads out instead of collapsing to a point.
const SPAN_FLOOR_S: f64 = 0.020;
/// Upper bound (model-seconds) on the displayed spike-time span. Sits just above the
/// clip-enforced worst-case latency `LAMBDA * MAX_SURPRISAL_BITS = 0.266 s`, so the span always
/// contains the slowest finite spike while keeping the fast winner visible.
const SPAN_CEIL_S: f64 = 0.30;

/// Replay duration for the current per-symbol hold: capped, and never more than half the hold
/// so there is always time to show the revealed result before the next symbol is latched.
fn race_secs(hold_secs: f64) -> f64 {
    (hold_secs * 0.5).min(RACE_SECS_CAP)
}

/// Live state of the spike-latency view (held `#[serde(skip)]` on the app).
pub struct SpikeState {
    /// The predict-and-learn agent (same coder as the gym).
    pub agent: DeltaAgent,
    /// The online momentum-rover source.
    pub env: OnlineRover,
    /// Whether the view is stepping.
    pub running: bool,
    /// Background learning rate (symbols / second). The learner keeps converging underneath
    /// while the display replays one sampled symbol at a time, so each replayed race is a
    /// little sharper than the last.
    pub learn_rate: f64,
    /// Wall-clock seconds each sampled symbol is shown (slow-motion race, then a held reveal).
    pub hold_secs: f64,
    /// Total symbols emitted.
    pub step_count: u64,
    // --- live (fast background) prediction, updated every micro-step ---
    q: Vec<f64>,
    emitted: usize,
    decoded: Option<usize>,
    // --- showcase (the readable, latched symbol currently on screen) ---
    show_q: Vec<f64>,
    show_emitted: usize,
    show_decoded: Option<usize>,
    /// Wall-clock seconds elapsed in the current showcased symbol.
    phase_t: f64,
    /// Fractional-symbol accumulator so the background rate is honoured at any frame rate.
    step_acc: f64,
    bits_window: VecDeque<f64>,
    acc_window: VecDeque<bool>,
}

impl Default for SpikeState {
    fn default() -> Self {
        Self {
            agent: DeltaAgent::paper(scenario::N),
            env: OnlineRover::new(Scenario::MarkovRover.source(7), LAMBDA, 7),
            running: true,
            learn_rate: 250.0,
            hold_secs: 2.2,
            step_count: 0,
            q: vec![0.25; scenario::N],
            emitted: 0,
            decoded: None,
            show_q: vec![0.25; scenario::N],
            show_emitted: 0,
            show_decoded: None,
            phase_t: 2.2,
            step_acc: 0.0,
            bits_window: VecDeque::new(),
            acc_window: VecDeque::new(),
        }
    }
}

impl SpikeState {
    fn step(&mut self) {
        let obs = self.env.context();
        let pred = self.agent.act(&obs);
        let out = self.env.step(&pred);
        let y = one_hot(out.emitted, self.env.n());
        self.agent.learn(&obs, &y, &pred.q);
        self.step_count += 1;
        self.q = pred.q;
        self.emitted = out.emitted;
        self.decoded = pred.decoded;

        self.bits_window.push_back(out.bits);
        if self.bits_window.len() > WINDOW {
            self.bits_window.pop_front();
        }
        self.acc_window.push_back(out.correct);
        if self.acc_window.len() > WINDOW {
            self.acc_window.pop_front();
        }
    }

    fn mean_bits(&self) -> f64 {
        if self.bits_window.is_empty() {
            f64::NAN
        } else {
            self.bits_window.iter().sum::<f64>() / self.bits_window.len() as f64
        }
    }

    fn accuracy(&self) -> f64 {
        if self.acc_window.is_empty() {
            f64::NAN
        } else {
            self.acc_window.iter().filter(|&&c| c).count() as f64 / self.acc_window.len() as f64
        }
    }

    /// Latch the current (fast, live) prediction as the symbol now on display and restart its
    /// slow-motion replay clock.
    fn latch_showcase(&mut self) {
        self.show_q = self.q.clone();
        self.show_emitted = self.emitted;
        self.show_decoded = self.decoded;
        self.phase_t = 0.0;
    }

    fn reset(&mut self) {
        let running = self.running;
        let learn_rate = self.learn_rate;
        let hold_secs = self.hold_secs;
        *self = Self::default();
        self.running = running;
        self.learn_rate = learn_rate;
        self.hold_secs = hold_secs;
        self.phase_t = self.hold_secs;
    }
}

/// The drive `J` and first-spike time for a predicted probability `q`. `q` is clamped to the
/// encoder's representable range `[Q_CLIP_LO, Q_CLIP_HI]`, so the latency obeys the same
/// clip-enforced worst case (`<= 0.266 s`) as the bits and can't run off to infinity.
fn drive_and_spike(q: f64) -> (f64, f64) {
    let q = q.clamp(Q_CLIP_LO, Q_CLIP_HI);
    let raw = calibration_drive(q, LAMBDA, TAU_RC, THETA);
    let j = if raw.is_finite() {
        raw.clamp(0.0, RHEOBASE_CEILING)
    } else {
        RHEOBASE_CEILING
    };
    (j, nengo_first_spike_time(j, TAU_RC))
}

/// The central spike-latency view: the four-neuron race to threshold.
pub fn spikes_view(app: &mut TemplateApp, ui: &mut egui::Ui, _ctx: &egui::Context) {
    if app.spikes.running {
        // The learner runs FAST in the background (so it converges in seconds), but the
        // DISPLAY runs SLOW: we latch one sampled symbol and replay its race over ~1 s, hold
        // the result, then advance. Otherwise the banner/bars would flip through thousands of
        // symbols a second and read as noise.
        //
        // Time-scale by `unstable_dt` (TRUE elapsed time): in reactive repaint mode
        // `stable_dt` is pinned to the refresh interval, so mouse-driven repaints would
        // over-count and the sim would speed up. A fractional-symbol accumulator keeps the
        // background rate exact at any frame rate.
        let dt = f64::from(ui.ctx().input(|i| i.unstable_dt)).clamp(0.0, 0.1);
        app.spikes.step_acc += app.spikes.learn_rate * dt;
        let steps = app.spikes.step_acc.floor();
        app.spikes.step_acc -= steps;
        for _ in 0..steps as usize {
            app.spikes.step();
        }
        // advance the slow display clock; when a symbol's time is up, latch the next one,
        // carrying the overshoot so the display tempo stays true to `hold_secs`
        app.spikes.phase_t += dt;
        if app.spikes.phase_t >= app.spikes.hold_secs {
            let overshoot = app.spikes.phase_t - app.spikes.hold_secs;
            app.spikes.latch_showcase();
            app.spikes.phase_t = overshoot.min(app.spikes.hold_secs);
        }
        // ~30 FPS so the slow-motion race animates smoothly
        ui.ctx().request_repaint_after(Duration::from_millis(33));
    }

    ui.heading("Spike latency — the wait until firing IS the surprise");
    ui.label(
        RichText::new(
            "Each symbol is a neuron. The more the coder expects a symbol (higher q), the \
             harder its neuron is driven, so the sooner it fires. The first to fire is the \
             coder's guess; the wait until it fires equals the code length -log2 q (bits). \
             Expected -> fires fast -> cheap;  surprising -> fires late -> expensive.",
        )
        .weak(),
    );
    ui.separator();

    // Render the latched showcase symbol (NOT the racing background one).
    let q = app.spikes.show_q.clone();
    let races: Vec<(f64, f64)> = q.iter().map(|&qi| drive_and_spike(qi)).collect();

    // Display span (model-seconds) covering every finite spike, shared by the cursor sweep, the
    // latency bars, and the voltage plot so all three agree and the plot x-axis never jumps on
    // reveal. Bounded below so an all-fast symbol still spreads, and above so a near-rheobase
    // straggler can't compress the fast winner into a sliver.
    let max_finite = races
        .iter()
        .map(|&(_, t)| t)
        .filter(|t| t.is_finite())
        .fold(0.0_f64, f64::max);
    let span_s = max_finite.clamp(SPAN_FLOOR_S, SPAN_CEIL_S);

    // Animation cursor (model-seconds). `Some(c)` while the race is replaying; `None` once it
    // is fully revealed (held), or whenever paused (so a pause always shows the full result).
    // The cursor sweeps the whole span, so every finite dot crosses in firing order.
    let race = race_secs(app.spikes.hold_secs);
    let cursor = if app.spikes.running && app.spikes.phase_t < race {
        Some((app.spikes.phase_t / race) * span_s)
    } else {
        None
    };
    // First spike (model-seconds): once the cursor passes it the guess is known → reveal banner.
    let first_spike = races
        .iter()
        .map(|&(_, t)| t)
        .filter(|t| t.is_finite())
        .fold(f64::INFINITY, f64::min);
    let revealed = cursor.is_none_or(|c| c >= first_spike);

    result_banner(
        ui,
        &q,
        &races,
        app.spikes.show_decoded,
        app.spikes.show_emitted,
        revealed,
    );
    ui.add_space(10.0);
    ui.label(
        RichText::new("time until each neuron fires  —  shorter = more expected = fewer bits")
            .small()
            .weak(),
    );
    latency_bars(
        ui,
        &q,
        &races,
        app.spikes.show_decoded,
        app.spikes.show_emitted,
        cursor,
        span_s,
    );
    ui.add_space(14.0);

    ui.label(
        RichText::new(format!(
            "Under the hood: each neuron's membrane ramps to the threshold θ; a stronger \
             drive (higher q) reaches it sooner. The first dot to touch θ is the guess. \
             Replayed in slow motion (~{race:.1} s) — the x-axis still shows the true model \
             time, where the spikes actually land within a few ms.",
        ))
        .small()
        .weak(),
    );

    spike_race_plot(ui, &q, &races, cursor, span_s);
    ui.label(
        RichText::new("x = time (ms); first dot to reach θ = the coder's guess")
            .small()
            .weak(),
    );
}

/// The "under the hood" membrane race: each neuron's ramp to threshold θ, animated up to the
/// moving `cursor` (model-seconds) so the dots cross in firing order. `span_s` is the shared
/// display span; locking the x-axis to it keeps the plot from zooming as the cursor sweeps.
fn spike_race_plot(
    ui: &mut egui::Ui,
    q: &[f64],
    races: &[(f64, f64)],
    cursor: Option<f64>,
    span_s: f64,
) {
    Plot::new("spike_race")
        .height(200.0)
        .legend(Legend::default())
        .include_y(0.0)
        .include_y(THETA * 1.15)
        .include_x(0.0)
        .include_x(span_s * 1000.0)
        .show(ui, |pui| {
            pui.hline(HLine::new("threshold θ", THETA).color(Color32::GRAY));
            if let Some(c) = cursor {
                pui.vline(
                    VLine::new("now", c * 1000.0).color(Color32::from_gray(90).gamma_multiply(0.7)),
                );
            }
            for (j, &(drive, t_spike)) in races.iter().enumerate() {
                let color = NEURON_COLORS.get(j).copied().unwrap_or(Color32::WHITE);
                let label = LABELS.get(j).copied().unwrap_or("?");
                // ramp end = spike time (or span end if it never fires), clipped to the cursor
                let t_full = if t_spike.is_finite() { t_spike } else { span_s };
                let t_end = cursor.map_or(t_full, |c| t_full.min(c));
                // membrane ramp V(t) = J (1 - e^{-t/tau})
                let pts: Vec<[f64; 2]> = (0..=40)
                    .map(|k| {
                        let t = t_end * f64::from(k) / 40.0;
                        let v = drive * (1.0 - (-t / TAU_RC).exp());
                        [t * 1000.0, v.min(THETA * 1.15)]
                    })
                    .collect();
                pui.line(
                    Line::new(
                        format!("{label}  q={:.2}", q.get(j).copied().unwrap_or(0.0)),
                        PlotPoints::from(pts),
                    )
                    .color(color),
                );
                // the spike dot appears only once the cursor has reached this neuron's spike
                let fired = t_spike.is_finite() && cursor.is_none_or(|c| c >= t_spike);
                if fired {
                    pui.points(
                        Points::new("", PlotPoints::from(vec![[t_spike * 1000.0, THETA]]))
                            .color(color)
                            .radius(5.0),
                    );
                }
            }
        });
}

/// The plain-language punchline: what the coder guessed, how fast / cheap, vs what came.
fn result_banner(
    ui: &mut egui::Ui,
    q: &[f64],
    races: &[(f64, f64)],
    decoded: Option<usize>,
    emitted: usize,
    revealed: bool,
) {
    let actual = LABELS.get(emitted).copied().unwrap_or("?");
    if !revealed {
        // mid-race: the guess is not known until the first neuron fires
        egui::Frame::new()
            .fill(Color32::from_gray(70).gamma_multiply(0.16))
            .inner_margin(8.0)
            .corner_radius(4.0)
            .show(ui, |ui| {
                ui.label(
                    RichText::new(
                        "The four neurons are racing — the first one to fire is the coder's guess…",
                    )
                    .size(15.0)
                    .strong()
                    .color(Color32::from_gray(190)),
                );
            });
        return;
    }
    let (text, color) = match decoded {
        Some(e) => {
            let guess = LABELS.get(e).copied().unwrap_or("?");
            let qe = q.get(e).copied().unwrap_or(0.0);
            let t_ms = races.get(e).map_or(f64::INFINITY, |&(_, t)| t) * 1000.0;
            let bits = -qe.clamp(Q_CLIP_LO, Q_CLIP_HI).log2();
            if Some(emitted) == decoded {
                (
                    format!(
                        "Guessed {guess} (q={qe:.2}) — fired in {t_ms:.1} ms = {bits:.2} bits.   \
                         Actual: {actual}.   MATCHED — paid only {bits:.2} bits."
                    ),
                    GREEN,
                )
            } else {
                let abits = -q
                    .get(emitted)
                    .copied()
                    .unwrap_or(1.0)
                    .clamp(Q_CLIP_LO, Q_CLIP_HI)
                    .log2();
                (
                    format!(
                        "Guessed {guess} (q={qe:.2}).   Actual: {actual}.   \
                         SURPRISED — the true symbol cost {abits:.2} bits."
                    ),
                    RED,
                )
            }
        }
        None => ("(no neuron fired — silent)".to_owned(), Color32::GRAY),
    };
    egui::Frame::new()
        .fill(color.gamma_multiply(0.16))
        .inner_margin(8.0)
        .corner_radius(4.0)
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(15.0).strong().color(color));
        });
}

/// Horizontal "time-to-fire" bars: bar length = first-spike latency (= surprise). The
/// shortest (first to fire) is the guess; the actual emitted symbol's row is highlighted.
///
/// `cursor` (model-seconds, `Some` while the race replays) grows each bar up to the moving
/// "now" line and only reveals a neuron's latency once it has fired — so the bars fill in the
/// same firing order as the voltage race. `None` shows every bar at its final length. `span_s`
/// is the shared display span used as the full-width denominator (matching the voltage plot).
fn latency_bars(
    ui: &mut egui::Ui,
    q: &[f64],
    races: &[(f64, f64)],
    decoded: Option<usize>,
    emitted: usize,
    cursor: Option<f64>,
    span_s: f64,
) {
    let n = races.len();
    let row_h = 28.0_f32;
    let width = ui.available_width().min(680.0);
    let (rect, _resp) =
        ui.allocate_exact_size(egui::vec2(width, n as f32 * row_h), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let max_t = span_s.max(1e-3);
    let label_w = 92.0_f32;
    let info_w = 196.0_f32;
    let bar_x0 = rect.left() + label_w;
    let bar_max = (rect.width() - label_w - info_w).max(20.0);
    let info_x = bar_x0 + bar_max + 8.0;
    for (j, &(_, t_spike)) in races.iter().enumerate() {
        let y = rect.top() + j as f32 * row_h;
        let cy = y + row_h * 0.5;
        let color = NEURON_COLORS.get(j).copied().unwrap_or(Color32::WHITE);
        let label = LABELS.get(j).copied().unwrap_or("?");
        let qi = q.get(j).copied().unwrap_or(0.0);

        // has this neuron fired yet (at the current cursor position)?
        let fired = t_spike.is_finite() && cursor.is_none_or(|c| c >= t_spike);
        // bar length in model-seconds: grows with the cursor, locks at the spike time
        let bar_t = match cursor {
            None => {
                if t_spike.is_finite() {
                    t_spike
                } else {
                    max_t
                }
            }
            Some(c) => {
                if t_spike.is_finite() {
                    t_spike.min(c)
                } else {
                    c.min(max_t)
                }
            }
        };

        // highlight the row of the symbol that was actually emitted
        if j == emitted {
            painter.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(rect.left(), y + 1.0),
                    egui::vec2(rect.width(), row_h - 2.0),
                ),
                3.0,
                Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 26),
            );
        }
        painter.text(
            egui::pos2(rect.left() + 2.0, cy),
            egui::Align2::LEFT_CENTER,
            format!("{label}  q={qi:.2}"),
            egui::FontId::monospace(13.0),
            color,
        );
        let frac = ((bar_t / max_t) as f32).clamp(0.0, 1.0).max(0.012);
        // dim while still climbing, full colour once fired (or never-fires, when revealed)
        let bar_color = if fired || (!t_spike.is_finite() && cursor.is_none()) {
            color
        } else {
            color.gamma_multiply(0.4)
        };
        painter.rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(bar_x0, y + 5.0),
                egui::vec2(bar_max * frac, row_h - 10.0),
            ),
            3.0,
            bar_color,
        );
        // reveal the latency text only after the neuron fires (keeps the order suspenseful)
        let info = if fired {
            let bits = -qi.clamp(Q_CLIP_LO, Q_CLIP_HI).log2();
            let mut s = format!("{:.1} ms = {bits:.2} bits", t_spike * 1000.0);
            if Some(j) == decoded {
                s.push_str("   1st");
            }
            if j == emitted {
                s.push_str(" · ACTUAL");
            }
            s
        } else if cursor.is_none() {
            // revealed but never fired
            let mut s = "never · inf bits".to_owned();
            if j == emitted {
                s.push_str(" · ACTUAL");
            }
            s
        } else {
            "…".to_owned()
        };
        painter.text(
            egui::pos2(info_x, cy),
            egui::Align2::LEFT_CENTER,
            info,
            egui::FontId::monospace(12.0),
            Color32::from_gray(215),
        );
    }
}

/// The right-panel inspector: run controls + live stats.
pub fn spikes_inspector(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.heading("Spike latency");
    ui.label(RichText::new("first-spike-takes-all decoder").weak());
    ui.separator();

    ui.horizontal(|ui| {
        let run_label = if app.spikes.running {
            "⏸ Pause"
        } else {
            "▶ Run"
        };
        if ui.button(run_label).clicked() {
            app.spikes.running = !app.spikes.running;
        }
        if ui.button("↺ Reset").clicked() {
            app.spikes.reset();
        }
        if ui.button("⏭ Step").clicked() {
            // one symbol, shown fully revealed (no replay) so you can inspect it. Pause too: a
            // manual step implies stopping, and it stops the same-frame auto-advance (the
            // inspector renders before the view) from immediately stomping this latched symbol.
            app.spikes.running = false;
            app.spikes.step();
            app.spikes.latch_showcase();
            app.spikes.phase_t = app.spikes.hold_secs;
        }
    });

    ui.add(
        egui::Slider::new(&mut app.spikes.hold_secs, 0.8..=5.0)
            .text("seconds per symbol (display tempo)"),
    );
    ui.add(
        egui::Slider::new(&mut app.spikes.learn_rate, 1.0..=5000.0)
            .text("background learning (symbols / s)")
            .logarithmic(true),
    );

    ui.separator();
    ui.label(RichText::new("Live stats").strong());
    egui::Grid::new("spike_stats").striped(true).show(ui, |ui| {
        row(ui, "symbols seen", format!("{}", app.spikes.step_count));
        row(ui, "bits/symbol", format!("{:.4}", app.spikes.mean_bits()));
        row(ui, "accuracy", format!("{:.4}", app.spikes.accuracy()));
        // show the latched (on-screen) symbol, not the fast background one, so it stays readable
        row(
            ui,
            "decoded",
            app.spikes
                .show_decoded
                .and_then(|d| LABELS.get(d).copied())
                .unwrap_or("-")
                .to_owned(),
        );
        row(
            ui,
            "actual",
            LABELS
                .get(app.spikes.show_emitted)
                .copied()
                .unwrap_or("-")
                .to_owned(),
        );
    });

    ui.separator();
    ui.label(
        RichText::new(
            "At the start every q is ~1/4, so the four neurons fire almost together — a \
             tie pays log2 4 = 2 bits (no compression). As the rover's momentum is learned, \
             the expected move's neuron fires earlier and the code shortens toward the \
             entropy rate. Use Step to advance one symbol at a time.",
        )
        .weak()
        .small(),
    );
}

fn row(ui: &mut egui::Ui, name: &str, value: String) {
    ui.label(name);
    ui.label(RichText::new(value).monospace());
    ui.end_row();
}
