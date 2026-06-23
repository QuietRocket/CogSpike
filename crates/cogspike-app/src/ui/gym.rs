//! The interactive rover gym view: a live "money plot" of bits/symbol descending
//! to the entropy-rate floor as the local delta rule learns `q -> P`, plus an
//! accuracy curve, learned-vs-true heatmaps, an editable weight matrix, and the
//! parity provenance badge.
//!
//! The environment is a pure frame-stepped object (no threads, no OS entropy), so
//! it runs identically on native and WASM.

use std::collections::VecDeque;

use cog_spike::coder::{
    self, LABELS, MAX_SURPRISAL_BITS, Q_CLIP_HI, Q_CLIP_LO, entropy_bits, entropy_rate, one_hot,
};
use cog_spike::gym::scenario::{self, Scenario};
use cog_spike::gym::{Agent as _, DeltaAgent, OnlineRover};
use cog_spike::substrate::{Parity, Substrate as _};
use egui_plot::{HLine, Legend, Line, Plot, PlotPoints, VLine};

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
    /// Short EMA of per-symbol surprise; the Boredom-Meter needle reads this.
    inst_ema: f64,
    /// `step_count`s at which the world was flipped (drawn as "world changed" markers).
    flip_marks: Vec<f64>,
    /// Whether the last flip put the world into its memoryless (s = 0) regime.
    flipped: bool,
    /// A twin agent with `eta0 = 0` (never learns) -- the control baseline. Run on the
    /// same stream, it stays at `log2(N) = 2` bits, so the gap to the learner controls
    /// for source difficulty (the learner's gain isn't luck).
    frozen: DeltaAgent,
    frozen_window: VecDeque<f64>,
    frozen_sum: f64,
    frozen_curve: Vec<[f64; 2]>,
    /// Cumulative bits the learner has saved over the frozen twin (the odometer).
    bits_saved: f64,
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
            inst_ema: (scenario::N as f64).log2(),
            flip_marks: Vec::new(),
            flipped: false,
            frozen: {
                let mut a = DeltaAgent::paper(scenario::N);
                a.eta0 = 0.0; // never learns
                a
            },
            frozen_window: VecDeque::new(),
            frozen_sum: 0.0,
            frozen_curve: Vec::new(),
            bits_saved: 0.0,
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

    /// Change the hidden world WITHOUT resetting the agent, so it must notice and
    /// re-learn. Toggles the rover between its sticky regime (`s = S`) and a memoryless
    /// one (`s = 0`); the learned `q` (tuned to the old regime) is now wrong, surprise
    /// spikes, and the floor moves to the new world's entropy rate. A regime change also
    /// re-opens plasticity (surprise-gated learning, à la Pearce-Hall) by resetting the
    /// learning-rate clock so re-adaptation is visible.
    fn flip_world(&mut self) {
        self.flipped = !self.flipped;
        let new_s = if self.flipped { 0.0 } else { coder::S };
        self.env.set_s(new_s);
        self.stickiness = new_s;
        self.agent.t = 0;
        self.flip_marks.push(self.step_count as f64);
    }

    fn micro_step(&mut self) {
        let obs = self.env.context();
        let pred = self.agent.act(&obs);
        let step = self.env.step(&pred);
        let y = one_hot(step.emitted, self.env.n());
        self.agent.learn(&obs, &y, &pred.q);
        self.step_count += 1;

        // Instantaneous surprise for the Boredom-Meter needle: a short EMA (effective
        // window ~20 symbols) so it spikes the moment the world changes, unlike the
        // 4000-symbol mean which only crawls.
        self.inst_ema += 0.05 * (step.bits - self.inst_ema);

        // Frozen twin: same context, never learns. It scores the same emitted symbol so
        // the running gap is a like-for-like control. (It does NOT call `learn`.)
        let frozen_q = self
            .frozen
            .act(&obs)
            .q
            .get(step.emitted)
            .copied()
            .unwrap_or(0.0)
            .clamp(Q_CLIP_LO, Q_CLIP_HI);
        let frozen_bits = -frozen_q.log2();
        self.bits_saved += frozen_bits - step.bits;
        self.frozen_window.push_back(frozen_bits);
        self.frozen_sum += frozen_bits;
        if self.frozen_window.len() > WINDOW {
            if let Some(old) = self.frozen_window.pop_front() {
                self.frozen_sum -= old;
            }
        }

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

    /// Advance `steps` micro-steps and append a plot sample. `steps` is computed from
    /// real elapsed time (not the repaint count) so the sim runs at a fixed wall-clock
    /// rate regardless of frame rate -- e.g. moving the mouse no longer speeds it up.
    fn advance(&mut self, steps: usize) {
        for _ in 0..steps {
            self.micro_step();
        }
        let x = self.step_count as f64;
        self.bits_curve.push([x, self.windowed_bits()]);
        self.acc_curve.push([x, self.windowed_acc()]);
        self.frozen_curve.push([x, self.windowed_frozen_bits()]);
        if self.bits_curve.len() > MAX_CURVE {
            self.bits_curve.remove(0);
        }
        if self.acc_curve.len() > MAX_CURVE {
            self.acc_curve.remove(0);
        }
        if self.frozen_curve.len() > MAX_CURVE {
            self.frozen_curve.remove(0);
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

    fn windowed_frozen_bits(&self) -> f64 {
        if self.frozen_window.is_empty() {
            f64::NAN
        } else {
            self.frozen_sum / self.frozen_window.len() as f64
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
        // Step by real elapsed time, not per repaint, so the sim runs at a fixed
        // wall-clock rate. `steps_per_frame` is the budget at 60 FPS; extra repaints
        // (e.g. from mouse movement) no longer accelerate the simulation. Use
        // `unstable_dt` (TRUE elapsed time) -- `stable_dt` can lock to the refresh rate.
        let dt = f64::from(ui.ctx().input(|i| i.unstable_dt)).clamp(0.0, 0.1);
        let steps = ((app.gym.steps_per_frame as f64) * 60.0 * dt).round() as usize;
        app.gym.advance(steps.max(1));
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
    let frozen_curve = app.gym.frozen_curve.clone();
    let flip_marks = app.gym.flip_marks.clone();

    // The Boredom Meter: instantaneous surprise vs the entropy-rate floor, fenced by
    // the clip-enforced cap. It parks at BORED once the agent has mastered the world.
    let time = ui.input(|i| i.time);
    surprise_needle(ui, app.gym.inst_ema, floor, *MAX_SURPRISAL_BITS, time);
    ui.add_space(6.0);

    Plot::new("bits_plot")
        .height(200.0)
        .legend(Legend::default())
        .show(ui, |pui| {
            pui.line(
                Line::new("frozen (no learning)", PlotPoints::from(frozen_curve))
                    .color(egui::Color32::from_rgb(150, 150, 160)),
            );
            pui.line(
                Line::new("learning agent", PlotPoints::from(bits_curve))
                    .color(egui::Color32::from_rgb(90, 170, 255)),
            );
            pui.hline(
                HLine::new("entropy-rate floor H(P)", floor)
                    .color(egui::Color32::from_rgb(64, 200, 120)),
            );
            pui.hline(HLine::new("marginal H(π)", marginal).color(egui::Color32::GRAY));
            for &mark in &flip_marks {
                pui.vline(
                    VLine::new("world changed", mark)
                        .color(egui::Color32::from_rgb(230, 110, 90))
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
            }
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
            ui.label(egui::RichText::new("learned q — mastery surface").strong());
            ui.label(
                egui::RichText::new("blue = predicted (cheap) · red = surprising")
                    .weak()
                    .small(),
            );
            heatmap_mastery(ui, "q_heat", &law);
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

    flip_control(app, ui);

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
        stat(
            ui,
            "bits saved vs frozen",
            format!("{:.1} kb", app.gym.bits_saved / 1000.0),
        );
    });

    ui.separator();
    parity_badge(ui, app.gym.agent.substrate.parity());
}

/// The "Flip the world" control (only meaningful for scenarios with memory).
fn flip_control(app: &mut TemplateApp, ui: &mut egui::Ui) {
    if !app.gym.scenario.has_stickiness() {
        return;
    }
    ui.add_space(4.0);
    let flip_label = if app.gym.flipped {
        "Restore world (sticky)"
    } else {
        "Flip the world (memoryless)"
    };
    if ui
        .add(
            egui::Button::new(
                egui::RichText::new(flip_label)
                    .strong()
                    .color(egui::Color32::WHITE),
            )
            .fill(egui::Color32::from_rgb(150, 70, 80)),
        )
        .clicked()
    {
        app.gym.flip_world();
    }
    ui.label(
        egui::RichText::new(
            "change a hidden rule the agent was never told — watch the needle spike, \
             then watch it re-learn",
        )
        .weak()
        .small(),
    );
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

/// The Boredom-Meter gauge: instantaneous surprise (`bits`) on a `[0, cap]` scale with
/// the entropy-rate floor marked. Reads BORED (green) once surprise settles near the
/// floor, SURPRISED (pulsing red) when it spikes. The fill can never reach the right
/// wall: that wall is the clip-enforced cap (`-log2(q_min)` = bounded latency).
fn surprise_needle(ui: &mut egui::Ui, bits: f64, floor: f64, cap: f64, time: f64) {
    let h = 56.0_f32;
    let w = ui.available_width().min(620.0);
    let (rect, _resp) = ui.allocate_exact_size(egui::vec2(w, h), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(
        rect,
        egui::CornerRadius::same(4),
        egui::Color32::from_rgb(22, 24, 30),
    );

    let cap = cap.max(1e-6);
    let frac = (bits / cap).clamp(0.0, 1.0) as f32;
    let bored = (bits - floor).max(0.0) < 0.4;

    let fill_color = if bored {
        egui::Color32::from_rgb(70, 190, 120)
    } else {
        let pulse = (0.5 + 0.5 * (time * 6.0).sin()) as f32;
        let g = (60.0 + 50.0 * (1.0 - pulse)) as u8;
        egui::Color32::from_rgb(230, g, 70)
    };
    let fill = egui::Rect::from_min_size(rect.min, egui::vec2(rect.width() * frac, rect.height()));
    painter.rect_filled(fill, egui::CornerRadius::same(4), fill_color);

    // entropy-rate floor tick ("fully mastered" surprise level)
    let fx = rect.left() + rect.width() * (floor / cap).clamp(0.0, 1.0) as f32;
    painter.line_segment(
        [egui::pos2(fx, rect.top()), egui::pos2(fx, rect.bottom())],
        egui::Stroke::new(1.5, egui::Color32::from_rgb(140, 220, 170)),
    );

    let (word, word_color) = if bored {
        ("BORED", egui::Color32::from_rgb(150, 230, 180))
    } else {
        ("SURPRISED", egui::Color32::from_rgb(255, 190, 160))
    };
    painter.text(
        egui::pos2(rect.left() + 12.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        format!("{word}    {bits:.2} bits surprise"),
        egui::FontId::proportional(18.0),
        word_color,
    );
    painter.text(
        egui::pos2(rect.right() - 8.0, rect.top() + 8.0),
        egui::Align2::RIGHT_TOP,
        format!("clip-enforced cap {cap:.1} b · latency <= 0.266 s"),
        egui::FontId::monospace(10.0),
        egui::Color32::from_rgb(190, 130, 130),
    );
    painter.text(
        egui::pos2(fx + 3.0, rect.bottom() - 7.0),
        egui::Align2::LEFT_BOTTOM,
        "floor",
        egui::FontId::monospace(9.0),
        egui::Color32::from_rgb(140, 220, 170),
    );
}

/// Like [`heatmap`] but coloured by per-cell surprisal `-log2(q)` (hot = surprising,
/// cold = mastered) so the learned matrix reads as a world-model "mastery surface".
fn heatmap_mastery(ui: &mut egui::Ui, id: &str, m: &[Vec<f64>]) {
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
                    .rect_filled(rect, egui::CornerRadius::same(2), surprisal_color(v));
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    format!("{v:.2}"),
                    egui::FontId::monospace(11.0),
                    egui::Color32::from_gray(235),
                );
            }
            ui.end_row();
        }
    });
}

/// Map a probability `v` to a hot/cold colour by its surprisal `-log2(v)`: high `v`
/// (cheap, mastered) -> cold blue; low `v` (surprising) -> hot red. Normalised by
/// `log2(N) = 2` bits (the uniform-predictor surprise).
fn surprisal_color(v: f64) -> egui::Color32 {
    let bits = -v.clamp(1e-6, 1.0).log2();
    let t = (bits / 2.0).clamp(0.0, 1.0) as f32; // 0 = mastered, 1 = surprising
    let lerp = |a: f32, b: f32| (a + (b - a) * t) as u8;
    egui::Color32::from_rgb(lerp(50.0, 210.0), lerp(110.0, 70.0), lerp(210.0, 55.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learner_beats_frozen_baseline() {
        // D2 control: a frozen (eta0 = 0) twin runs on the SAME stream as the learner.
        let mut gym = GymState::new(Scenario::MarkovRover, 7);
        gym.advance(60_000);

        // The frozen twin never learns -> its surprise sits at exactly log2(N) = 2 bits.
        assert!(
            (gym.windowed_frozen_bits() - 2.0).abs() < 1e-9,
            "frozen baseline should be 2 bits, got {}",
            gym.windowed_frozen_bits()
        );
        // The learner compresses well below the 2-bit baseline...
        assert!(
            gym.windowed_bits() < 1.5,
            "learner should beat the baseline, got {}",
            gym.windowed_bits()
        );
        // ...so it banks a large, positive bit-saving over the same stream.
        assert!(
            gym.bits_saved > 5_000.0,
            "learner should save many bits vs frozen, got {}",
            gym.bits_saved
        );
    }
}
