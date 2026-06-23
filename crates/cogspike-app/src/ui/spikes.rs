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
    LABELS, LAMBDA, RHEOBASE_CEILING, TAU_RC, THETA, calibration_drive, nengo_first_spike_time,
    one_hot,
};
use cog_spike::gym::scenario::{self, Scenario};
use cog_spike::gym::{Agent as _, DeltaAgent, OnlineRover};
use egui::{Color32, RichText};
use egui_plot::{HLine, Legend, Line, Plot, PlotPoints, Points};

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

/// Live state of the spike-latency view (held `#[serde(skip)]` on the app).
pub struct SpikeState {
    /// The predict-and-learn agent (same coder as the gym).
    pub agent: DeltaAgent,
    /// The online momentum-rover source.
    pub env: OnlineRover,
    /// Whether the view is stepping.
    pub running: bool,
    /// Learning symbols processed per displayed frame (fast background convergence).
    pub steps_per_frame: usize,
    /// Total symbols emitted.
    pub step_count: u64,
    q: Vec<f64>,
    emitted: usize,
    decoded: Option<usize>,
    bits_window: VecDeque<f64>,
    acc_window: VecDeque<bool>,
}

impl Default for SpikeState {
    fn default() -> Self {
        Self {
            agent: DeltaAgent::paper(scenario::N),
            env: OnlineRover::new(Scenario::MarkovRover.source(7), LAMBDA, 7),
            running: true,
            steps_per_frame: 200,
            step_count: 0,
            q: vec![0.25; scenario::N],
            emitted: 0,
            decoded: None,
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

    fn reset(&mut self) {
        let running = self.running;
        *self = Self::default();
        self.running = running;
    }
}

/// The drive `J` and first-spike time for a predicted probability `q`.
fn drive_and_spike(q: f64) -> (f64, f64) {
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
        // Step by real elapsed time, not per repaint, so the learner converges at a fixed
        // wall-clock rate (mouse movement no longer accelerates it). Use `unstable_dt`
        // (TRUE elapsed time): in reactive repaint mode `stable_dt` is locked to the
        // refresh interval, so mouse-driven repaints would over-count. The view nominally
        // repaints every 80 ms (12.5 FPS); `steps_per_frame` is that per-frame budget.
        let dt = f64::from(ui.ctx().input(|i| i.unstable_dt)).clamp(0.0, 0.1);
        let steps = ((app.spikes.steps_per_frame as f64) * 12.5 * dt).round() as usize;
        for _ in 0..steps {
            app.spikes.step();
        }
        ui.ctx().request_repaint_after(Duration::from_millis(80));
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

    let q = app.spikes.q.clone();
    let races: Vec<(f64, f64)> = q.iter().map(|&qi| drive_and_spike(qi)).collect();

    result_banner(ui, &q, &races, app.spikes.decoded, app.spikes.emitted);
    ui.add_space(10.0);
    ui.label(
        RichText::new("time until each neuron fires  —  shorter = more expected = fewer bits")
            .small()
            .weak(),
    );
    latency_bars(ui, &q, &races, app.spikes.decoded, app.spikes.emitted);
    ui.add_space(14.0);

    ui.label(
        RichText::new(
            "Under the hood: each neuron's membrane ramps to the threshold θ; a stronger \
             drive (higher q) reaches it sooner. The first dot to touch θ is the guess.",
        )
        .small()
        .weak(),
    );
    let window_ms = races
        .iter()
        .map(|&(_, t)| t)
        .filter(|t| t.is_finite())
        .fold(0.0_f64, f64::max)
        .mul_add(1000.0 * 1.25, 0.0)
        .clamp(10.0, 80.0);

    Plot::new("spike_race")
        .height(200.0)
        .legend(Legend::default())
        .include_y(0.0)
        .include_y(THETA * 1.15)
        .show(ui, |pui| {
            pui.hline(HLine::new("threshold θ", THETA).color(Color32::GRAY));
            for (j, &(drive, t_spike)) in races.iter().enumerate() {
                let color = NEURON_COLORS.get(j).copied().unwrap_or(Color32::WHITE);
                let label = LABELS.get(j).copied().unwrap_or("?");
                // membrane ramp V(t) = J (1 - e^{-t/tau}), sampled to the spike or window end
                let t_end = if t_spike.is_finite() {
                    t_spike
                } else {
                    window_ms / 1000.0
                };
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
                if t_spike.is_finite() {
                    pui.points(
                        Points::new("", PlotPoints::from(vec![[t_spike * 1000.0, THETA]]))
                            .color(color)
                            .radius(5.0),
                    );
                }
            }
        });
    ui.label(
        RichText::new("x = time (ms); first dot to reach θ = the coder's guess")
            .small()
            .weak(),
    );
}

/// The plain-language punchline: what the coder guessed, how fast / cheap, vs what came.
fn result_banner(
    ui: &mut egui::Ui,
    q: &[f64],
    races: &[(f64, f64)],
    decoded: Option<usize>,
    emitted: usize,
) {
    let actual = LABELS.get(emitted).copied().unwrap_or("?");
    let (text, color) = match decoded {
        Some(e) => {
            let guess = LABELS.get(e).copied().unwrap_or("?");
            let qe = q.get(e).copied().unwrap_or(0.0);
            let t_ms = races.get(e).map_or(f64::INFINITY, |&(_, t)| t) * 1000.0;
            let bits = -qe.clamp(1e-4, 0.999).log2();
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
                    .clamp(1e-4, 0.999)
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
fn latency_bars(
    ui: &mut egui::Ui,
    q: &[f64],
    races: &[(f64, f64)],
    decoded: Option<usize>,
    emitted: usize,
) {
    let n = races.len();
    let row_h = 28.0_f32;
    let width = ui.available_width().min(680.0);
    let (rect, _resp) =
        ui.allocate_exact_size(egui::vec2(width, n as f32 * row_h), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let max_t = races
        .iter()
        .map(|&(_, t)| t)
        .filter(|t| t.is_finite())
        .fold(0.0_f64, f64::max)
        .max(1e-3);
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
        let (frac, info) = if t_spike.is_finite() {
            let bits = -qi.clamp(1e-4, 0.999).log2();
            (
                ((t_spike / max_t) as f32).clamp(0.03, 1.0),
                format!("{:.1} ms = {bits:.2} bits", t_spike * 1000.0),
            )
        } else {
            (1.0, "never · inf bits".to_owned())
        };
        let bar_color = if t_spike.is_finite() {
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
        let mut info = info;
        if Some(j) == decoded {
            info.push_str("   1st");
        }
        if j == emitted {
            info.push_str(" · ACTUAL");
        }
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
            app.spikes.step();
        }
    });

    ui.add(
        egui::Slider::new(&mut app.spikes.steps_per_frame, 1..=2000)
            .text("learn speed (symbols / frame)")
            .logarithmic(true),
    );

    ui.separator();
    ui.label(RichText::new("Live stats").strong());
    egui::Grid::new("spike_stats").striped(true).show(ui, |ui| {
        row(ui, "symbols seen", format!("{}", app.spikes.step_count));
        row(ui, "bits/symbol", format!("{:.4}", app.spikes.mean_bits()));
        row(ui, "accuracy", format!("{:.4}", app.spikes.accuracy()));
        row(
            ui,
            "decoded",
            app.spikes
                .decoded
                .and_then(|d| LABELS.get(d).copied())
                .unwrap_or("-")
                .to_owned(),
        );
        row(
            ui,
            "actual",
            LABELS
                .get(app.spikes.emitted)
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
