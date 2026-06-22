//! The interactive rover gym view: a live "money plot" of bits/symbol descending
//! to the entropy-rate floor as the local delta rule learns `q -> P`, plus an
//! accuracy curve, learned-vs-true heatmaps, an editable weight matrix, and the
//! parity provenance badge.
//!
//! The environment is a pure frame-stepped object (no threads, no OS entropy), so
//! it runs identically on native and WASM.

use std::collections::VecDeque;

use cog_spike::coder::{self, LABELS, entropy_bits, entropy_rate, one_hot};
use cog_spike::gym::scenario::{self, Scenario};
use cog_spike::gym::{Agent as _, DeltaAgent, OnlineRover};
use cog_spike::substrate::{Parity, Substrate as _};
use egui_plot::{HLine, Legend, Line, Plot, PlotPoints};

use crate::app::TemplateApp;

const WINDOW: usize = 4000; // running-window size for the live stats
const MAX_CURVE: usize = 2000; // max points kept per plot line

/// All live state of the playground gym (held `#[serde(skip)]` on the app).
pub struct GymState {
    /// The currently selected source scenario.
    pub scenario: Scenario,
    /// The predict-and-learn agent.
    pub agent: DeltaAgent,
    /// The online (unbounded) source environment.
    pub env: OnlineRover,
    /// Whether the simulation is advancing each frame.
    pub running: bool,
    /// Micro-steps advanced per UI frame.
    pub steps_per_frame: usize,
    /// Source stickiness `s` (live slider).
    pub stickiness: f64,
    /// RNG seed.
    pub seed: u64,
    /// Total micro-steps taken.
    pub step_count: u64,
    bits_window: VecDeque<f64>,
    bits_sum: f64,
    acc_window: VecDeque<bool>,
    acc_sum: usize,
    bits_curve: Vec<[f64; 2]>,
    acc_curve: Vec<[f64; 2]>,
}

impl Default for GymState {
    fn default() -> Self {
        Self::new(Scenario::MarkovRover, 7)
    }
}

impl GymState {
    fn new(sc: Scenario, seed: u64) -> Self {
        Self {
            scenario: sc,
            agent: DeltaAgent::paper(scenario::N),
            env: OnlineRover::new(sc.source(seed), coder::LAMBDA, seed),
            running: false,
            steps_per_frame: 2000,
            stickiness: coder::S,
            seed,
            step_count: 0,
            bits_window: VecDeque::new(),
            bits_sum: 0.0,
            acc_window: VecDeque::new(),
            acc_sum: 0,
            bits_curve: Vec::new(),
            acc_curve: Vec::new(),
        }
    }

    /// Reset agent + env + history, keeping the scenario, seed, stickiness, and pace.
    fn reset(&mut self) {
        let (sc, s, seed, spf, running) = (
            self.scenario,
            self.stickiness,
            self.seed,
            self.steps_per_frame,
            self.running,
        );
        *self = Self::new(sc, seed);
        self.steps_per_frame = spf;
        self.stickiness = s;
        self.running = running;
        if sc.has_stickiness() {
            self.env.set_s(s);
        }
    }

    /// Switch to a different source scenario (fresh agent + history).
    fn set_scenario(&mut self, sc: Scenario) {
        let (seed, spf, running) = (self.seed, self.steps_per_frame, self.running);
        *self = Self::new(sc, seed);
        self.steps_per_frame = spf;
        self.running = running;
    }

    fn micro_step(&mut self) {
        let obs = self.env.context();
        let pred = self.agent.act(&obs);
        let step = self.env.step(&pred);
        let y = one_hot(step.emitted, self.env.n());
        self.agent.learn(&obs, &y, &pred.q);
        self.step_count += 1;

        self.bits_window.push_back(step.bits);
        self.bits_sum += step.bits;
        if self.bits_window.len() > WINDOW {
            if let Some(old) = self.bits_window.pop_front() {
                self.bits_sum -= old;
            }
        }
        self.acc_window.push_back(step.correct);
        self.acc_sum += usize::from(step.correct);
        if self.acc_window.len() > WINDOW {
            if let Some(old) = self.acc_window.pop_front() {
                self.acc_sum -= usize::from(old);
            }
        }
    }

    /// Advance one frame's worth of micro-steps and append a plot sample.
    fn advance(&mut self) {
        for _ in 0..self.steps_per_frame {
            self.micro_step();
        }
        let x = self.step_count as f64;
        self.bits_curve.push([x, self.windowed_bits()]);
        self.acc_curve.push([x, self.windowed_acc()]);
        if self.bits_curve.len() > MAX_CURVE {
            self.bits_curve.remove(0);
        }
        if self.acc_curve.len() > MAX_CURVE {
            self.acc_curve.remove(0);
        }
    }

    fn windowed_bits(&self) -> f64 {
        if self.bits_window.is_empty() {
            f64::NAN
        } else {
            self.bits_sum / self.bits_window.len() as f64
        }
    }

    fn windowed_acc(&self) -> f64 {
        if self.acc_window.is_empty() {
            f64::NAN
        } else {
            self.acc_sum as f64 / self.acc_window.len() as f64
        }
    }

    fn floor(&self) -> f64 {
        entropy_rate(&self.env.source.p, &self.env.source.pi)
    }

    fn marginal(&self) -> f64 {
        entropy_bits(&self.env.source.pi)
    }

    fn bayes_ceiling(&self) -> f64 {
        self.env
            .source
            .p
            .iter()
            .zip(&self.env.source.pi)
            .map(|(row, &pii)| pii * row.iter().copied().fold(0.0_f64, f64::max))
            .sum()
    }

    fn max_abs_q_minus_p(&self) -> f64 {
        let law = self.agent.net.conditional_law();
        self.env
            .source
            .p
            .iter()
            .zip(&law)
            .flat_map(|(pr, qr)| pr.iter().zip(qr).map(|(&a, &b)| (a - b).abs()))
            .fold(0.0_f64, f64::max)
    }
}

/// The central gym view: money plot + accuracy curve + heatmaps + weight editor.
pub fn gym_view(app: &mut TemplateApp, ui: &mut egui::Ui, _ctx: &egui::Context) {
    if app.gym.running {
        app.gym.advance();
        ui.ctx().request_repaint();
    }

    let sc = app.gym.scenario;
    ui.horizontal(|ui| {
        ui.heading("Compression playground");
        ui.label(
            egui::RichText::new("first-spike latency = surprisal; the delta rule learns q -> P")
                .weak(),
        );
    });
    ui.label(
        egui::RichText::new(format!("Scenario: {} — {}", sc.label(), sc.tagline()))
            .italics()
            .color(egui::Color32::from_rgb(160, 190, 230)),
    );
    ui.separator();

    let floor = app.gym.floor();
    let marginal = app.gym.marginal();
    let ceiling = app.gym.bayes_ceiling();
    let bits_curve = app.gym.bits_curve.clone();
    let acc_curve = app.gym.acc_curve.clone();

    Plot::new("bits_plot")
        .height(200.0)
        .legend(Legend::default())
        .show(ui, |pui| {
            pui.line(
                Line::new("bits/symbol", PlotPoints::from(bits_curve))
                    .color(egui::Color32::from_rgb(90, 170, 255)),
            );
            pui.hline(
                HLine::new("entropy-rate floor H(P)", floor)
                    .color(egui::Color32::from_rgb(64, 200, 120)),
            );
            pui.hline(HLine::new("marginal H(π)", marginal).color(egui::Color32::GRAY));
        });

    Plot::new("acc_plot")
        .height(140.0)
        .legend(Legend::default())
        .show(ui, |pui| {
            pui.line(
                Line::new("accuracy", PlotPoints::from(acc_curve))
                    .color(egui::Color32::from_rgb(255, 170, 90)),
            );
            pui.hline(
                HLine::new("Bayes ceiling", ceiling).color(egui::Color32::from_rgb(64, 200, 120)),
            );
        });

    ui.separator();

    let law = app.gym.agent.net.conditional_law();
    let p_true = app.gym.env.source.p.clone();
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(egui::RichText::new("learned q = softmax(W)").strong());
            heatmap(ui, "q_heat", &law);
        });
        ui.add_space(20.0);
        ui.vertical(|ui| {
            ui.label(egui::RichText::new("true P").strong());
            heatmap(ui, "p_heat", &p_true);
        });
        ui.add_space(20.0);
        ui.vertical(|ui| {
            ui.label(egui::RichText::new("W (logits) — edit while paused").strong());
            weight_editor(ui, &mut app.gym.agent.net.w);
        });
    });
}

/// The right-panel inspector: run controls, parameters, live stats, parity badge.
pub fn gym_inspector(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.heading("Playground");
    ui.label(egui::RichText::new("idealized latency entropy coder").weak());
    ui.separator();

    ui.label(egui::RichText::new("Source scenario").strong());
    let mut sc = app.gym.scenario;
    egui::ComboBox::from_id_salt("scenario_select")
        .selected_text(sc.label())
        .show_ui(ui, |ui| {
            for option in Scenario::ALL {
                ui.selectable_value(&mut sc, option, option.label());
            }
        });
    if sc != app.gym.scenario {
        app.gym.set_scenario(sc);
    }

    ui.separator();
    ui.horizontal(|ui| {
        let run_label = if app.gym.running {
            "⏸ Pause"
        } else {
            "▶ Run"
        };
        if ui.button(run_label).clicked() {
            app.gym.running = !app.gym.running;
        }
        if ui.button("↺ Reset").clicked() {
            app.gym.reset();
        }
    });

    ui.separator();
    ui.label(egui::RichText::new("Parameters").strong());
    if app.gym.scenario.has_stickiness() {
        let mut s = app.gym.stickiness;
        if ui
            .add(egui::Slider::new(&mut s, 0.0..=0.95).text("stickiness s"))
            .changed()
        {
            app.gym.stickiness = s;
            app.gym.env.set_s(s);
        }
    }
    ui.add(
        egui::Slider::new(&mut app.gym.steps_per_frame, 100..=20_000)
            .text("steps / frame")
            .logarithmic(true),
    );
    ui.horizontal(|ui| {
        ui.label("seed");
        ui.add(egui::DragValue::new(&mut app.gym.seed));
        if ui.button("apply").clicked() {
            app.gym.reset();
        }
    });

    ui.separator();
    ui.label(egui::RichText::new("Live stats").strong());
    egui::Grid::new("gym_stats").striped(true).show(ui, |ui| {
        stat(ui, "step", format!("{}", app.gym.step_count));
        stat(ui, "bits/symbol", format!("{:.4}", app.gym.windowed_bits()));
        stat(ui, "floor H(P)", format!("{:.4}", app.gym.floor()));
        stat(ui, "accuracy", format!("{:.4}", app.gym.windowed_acc()));
        stat(
            ui,
            "Bayes ceiling",
            format!("{:.4}", app.gym.bayes_ceiling()),
        );
        stat(
            ui,
            "max|q − P|",
            format!("{:.4}", app.gym.max_abs_q_minus_p()),
        );
        stat(
            ui,
            "η (learning rate)",
            format!("{:.5}", app.gym.agent.eta()),
        );
    });

    ui.separator();
    parity_badge(ui, app.gym.agent.substrate.parity());
}

fn stat(ui: &mut egui::Ui, name: &str, value: String) {
    ui.label(name);
    ui.label(egui::RichText::new(value).monospace());
    ui.end_row();
}

fn parity_badge(ui: &mut egui::Ui, parity: Parity) {
    let (sym, text, color) = match parity {
        Parity::Exact { tol } => (
            "=",
            format!("exact · idealized rung · ±{tol:.0e}"),
            egui::Color32::from_rgb(64, 200, 120),
        ),
        Parity::Band { rel } => (
            "≈",
            format!("sampled band · ±{:.0}%", rel * 100.0),
            egui::Color32::from_rgb(230, 180, 60),
        ),
        Parity::Proven => (
            "⊢",
            "proven (PRISM/PCTL)".to_owned(),
            egui::Color32::from_rgb(120, 160, 250),
        ),
    };
    ui.label(
        egui::RichText::new(format!("{sym}  {text}"))
            .strong()
            .color(color),
    );
    ui.label(
        egui::RichText::new("≈ spiking rung (M5) · ⊢ PRISM verify (M5) — reserved")
            .weak()
            .small(),
    );
}

/// Draw an `n x n` probability matrix as a labelled colour heatmap.
fn heatmap(ui: &mut egui::Ui, id: &str, m: &[Vec<f64>]) {
    let cell = 36.0_f32;
    egui::Grid::new(id).spacing([3.0, 3.0]).show(ui, |ui| {
        ui.label("");
        for lab in LABELS {
            ui.label(egui::RichText::new(lab).strong());
        }
        ui.end_row();
        for (i, row) in m.iter().enumerate() {
            ui.label(egui::RichText::new(LABELS.get(i).copied().unwrap_or("?")).strong());
            for &v in row {
                let (rect, _resp) =
                    ui.allocate_exact_size(egui::vec2(cell, cell), egui::Sense::hover());
                ui.painter()
                    .rect_filled(rect, egui::CornerRadius::same(2), cell_color(v));
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    format!("{v:.2}"),
                    egui::FontId::monospace(11.0),
                    text_color(v),
                );
            }
            ui.end_row();
        }
    });
}

/// The editable logit matrix `W` (4x4 drag values).
fn weight_editor(ui: &mut egui::Ui, w: &mut [Vec<f64>]) {
    egui::Grid::new("w_editor")
        .spacing([3.0, 3.0])
        .show(ui, |ui| {
            ui.label("");
            for lab in LABELS {
                ui.label(egui::RichText::new(lab).strong());
            }
            ui.end_row();
            for (i, row) in w.iter_mut().enumerate() {
                ui.label(egui::RichText::new(LABELS.get(i).copied().unwrap_or("?")).strong());
                for wij in row.iter_mut() {
                    ui.add(
                        egui::DragValue::new(wij)
                            .speed(0.05)
                            .fixed_decimals(2)
                            .range(-12.0..=12.0),
                    );
                }
                ui.end_row();
            }
        });
}

/// Map a probability `v` in `[0, 1]` to a dark-blue -> bright-yellow heat colour.
fn cell_color(v: f64) -> egui::Color32 {
    let t = v.clamp(0.0, 1.0) as f32;
    let lerp = |a: f32, b: f32| (a + (b - a) * t) as u8;
    egui::Color32::from_rgb(lerp(24.0, 250.0), lerp(28.0, 226.0), lerp(70.0, 80.0))
}

fn text_color(v: f64) -> egui::Color32 {
    if v > 0.45 {
        egui::Color32::BLACK
    } else {
        egui::Color32::from_gray(220)
    }
}
