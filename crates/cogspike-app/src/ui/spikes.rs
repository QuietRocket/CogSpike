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
    LABELS, LAMBDA, RHEOBASE_CEILING, TAU_RC, THETA, analytic_latency_ideal, calibration_drive,
    nengo_first_spike_time, one_hot,
};
use cog_spike::gym::scenario::{self, Scenario};
use cog_spike::gym::{Agent as _, DeltaAgent, OnlineRover};
use egui::{Color32, RichText};
use egui_plot::{HLine, Legend, Line, Plot, PlotPoints, Points};

use crate::app::TemplateApp;

const WINDOW: usize = 2000;
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
        for _ in 0..app.spikes.steps_per_frame.max(1) {
            app.spikes.step();
        }
        ui.ctx().request_repaint_after(Duration::from_millis(80));
    }

    ui.heading("Spike latency — surprisal is the wait for the first spike");
    ui.label(
        RichText::new(
            "Each symbol is a neuron. Its membrane is driven so that the more the coder \
             EXPECTS that symbol (higher q), the harder it is driven and the SOONER it \
             fires. The first neuron to cross threshold is the prediction; the time it \
             takes is the code length t*(q) = -lambda log2 q. Watch the winner pull ahead \
             as the coder learns.",
        )
        .weak(),
    );
    ui.separator();

    let q = app.spikes.q.clone();
    let races: Vec<(f64, f64)> = q.iter().map(|&qi| drive_and_spike(qi)).collect();
    let window_ms = races
        .iter()
        .map(|&(_, t)| t)
        .filter(|t| t.is_finite())
        .fold(0.0_f64, f64::max)
        .mul_add(1000.0 * 1.25, 0.0)
        .clamp(10.0, 80.0);

    Plot::new("spike_race")
        .height(280.0)
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
    ui.label(RichText::new("x = time (ms) · earliest spike = decoded symbol · no crossing = effectively infinite surprise").small().weak());
    ui.separator();

    draw_table(ui, &q, &races, app.spikes.decoded, app.spikes.emitted);
}

/// The per-symbol breakdown: predicted `q`, first-spike latency (= surprisal), and tags.
fn draw_table(
    ui: &mut egui::Ui,
    q: &[f64],
    races: &[(f64, f64)],
    decoded: Option<usize>,
    emitted: usize,
) {
    egui::Grid::new("spike_table")
        .striped(true)
        .spacing([18.0, 4.0])
        .show(ui, |ui| {
            for h in ["symbol", "q (predicted)", "latency = surprisal", ""] {
                ui.label(RichText::new(h).strong());
            }
            ui.end_row();
            for (j, &(_, t_spike)) in races.iter().enumerate() {
                let label = LABELS.get(j).copied().unwrap_or("?");
                let color = NEURON_COLORS.get(j).copied().unwrap_or(Color32::WHITE);
                ui.label(RichText::new(label).strong().color(color));
                ui.label(format!("{:.3}", q.get(j).copied().unwrap_or(0.0)));
                if t_spike.is_finite() {
                    let bits =
                        analytic_latency_ideal(q.get(j).copied().unwrap_or(1.0), LAMBDA) / LAMBDA;
                    ui.label(format!("{:.1} ms   ({bits:.2} bits)", t_spike * 1000.0));
                } else {
                    ui.label(RichText::new("never (inf)").italics());
                }
                let mut tag = String::new();
                if Some(j) == decoded {
                    tag.push_str("fires first");
                }
                if j == emitted {
                    if !tag.is_empty() {
                        tag.push_str(" · ");
                    }
                    tag.push_str("actual");
                }
                ui.label(RichText::new(tag).color(Color32::from_rgb(80, 210, 130)));
                ui.end_row();
            }
        });
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
