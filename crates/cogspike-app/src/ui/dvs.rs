//! The event-camera (DVS) view: a moving bar, its raw ON/OFF event stream, and the
//! sparse residual that survives predictive-inhibitory subtraction -- a literal,
//! hardware-native compressor (spikes in, spikes out, residual = surprise).

use std::collections::VecDeque;
use std::time::Duration;

use cog_spike::dvs::{DVS_WINDOW, DvsFrame, DvsScene, P_PIXELS};
use egui::{Color32, RichText};

use crate::app::TemplateApp;

const ON_COLOR: Color32 = Color32::from_rgb(90, 200, 235);
const OFF_COLOR: Color32 = Color32::from_rgb(240, 150, 70);
const RASTER_BG: Color32 = Color32::from_rgb(18, 20, 26);
const GREEN: Color32 = Color32::from_rgb(80, 210, 130);

/// Live state of the event-camera view (held `#[serde(skip)]` on the app).
pub struct DvsState {
    /// The frame-stepped event camera + predictive compressor.
    pub scene: DvsScene,
    /// Whether the camera advances each UI frame.
    pub running: bool,
}

impl Default for DvsState {
    fn default() -> Self {
        Self {
            scene: DvsScene::new(7),
            running: true,
        }
    }
}

/// The central event-camera view: headline ratio, bar strip, raw + residual rasters.
pub fn dvs_view(app: &mut TemplateApp, ui: &mut egui::Ui, _ctx: &egui::Context) {
    if app.dvs.running {
        app.dvs.scene.step();
        ui.ctx().request_repaint_after(Duration::from_millis(70));
    }

    ui.heading("Event camera — predictive compression");
    ui.label(
        RichText::new(
            "A bright bar drifts across a pixel strip. Each pixel fires an ON spike when \
             the bar arrives and an OFF spike when it leaves. A predictor anticipates the \
             next frame's spikes from the bar's motion and cancels them -- only the \
             UNPREDICTED spikes survive. The residual IS the compressed video.",
        )
        .weak(),
    );
    ui.separator();

    let ratio = app.dvs.scene.ratio();
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("{ratio:.1}×"))
                .heading()
                .strong()
                .color(GREEN),
        );
        ui.label(
            RichText::new(format!(
                "compression   ({} raw events  ->  {} residual spikes)",
                app.dvs.scene.raw_total, app.dvs.scene.resid_total
            ))
            .weak(),
        );
    });
    ui.add_space(8.0);

    ui.label(RichText::new("bar position (brightness)").small().weak());
    brightness_strip(ui, app.dvs.scene.current_brightness());
    ui.add_space(10.0);

    ui.label(RichText::new("RAW event stream — every edge, frame by frame").strong());
    raster(ui, &app.dvs.scene.frames, false);
    ui.add_space(10.0);

    ui.label(
        RichText::new(format!(
            "RESIDUAL after predictive subtraction — {ratio:.1}× sparser"
        ))
        .strong()
        .color(GREEN),
    );
    raster(ui, &app.dvs.scene.frames, true);
    ui.add_space(8.0);
    legend(ui);
}

/// The right-panel inspector: run controls, motion sliders, live stats.
pub fn dvs_inspector(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.heading("Event camera");
    ui.label(RichText::new("predictive-inhibitory DVS compressor").weak());
    ui.separator();

    ui.horizontal(|ui| {
        let run_label = if app.dvs.running {
            "⏸ Pause"
        } else {
            "▶ Run"
        };
        if ui.button(run_label).clicked() {
            app.dvs.running = !app.dvs.running;
        }
        if ui.button("↺ Reset").clicked() {
            let (v, j, pred) = (
                app.dvs.scene.velocity,
                app.dvs.scene.jitter,
                app.dvs.scene.predictor_on,
            );
            app.dvs.scene = DvsScene::new(7);
            app.dvs.scene.velocity = v;
            app.dvs.scene.jitter = j;
            app.dvs.scene.predictor_on = pred;
        }
    });

    ui.separator();
    ui.label(RichText::new("Motion").strong());
    ui.add(egui::Slider::new(&mut app.dvs.scene.velocity, -3..=3).text("bar velocity (px/frame)"));
    ui.add(egui::Slider::new(&mut app.dvs.scene.jitter, 0.0..=1.0).text("velocity jitter"));
    ui.checkbox(&mut app.dvs.scene.predictor_on, "predictor on (compress)");

    ui.separator();
    ui.label(RichText::new("Live stats").strong());
    egui::Grid::new("dvs_stats").striped(true).show(ui, |ui| {
        stat(ui, "raw events", format!("{}", app.dvs.scene.raw_total));
        stat(
            ui,
            "residual spikes",
            format!("{}", app.dvs.scene.resid_total),
        );
        stat(ui, "compression", format!("{:.2}×", app.dvs.scene.ratio()));
    });

    ui.separator();
    ui.label(
        RichText::new(
            "Turn the predictor OFF to see the raw stream pass through uncompressed. \
             Raise the jitter to make the motion unpredictable and watch compression \
             collapse toward 1× — compression tracks predictability.",
        )
        .weak()
        .small(),
    );
}

fn stat(ui: &mut egui::Ui, name: &str, value: String) {
    ui.label(name);
    ui.label(RichText::new(value).monospace());
    ui.end_row();
}

/// Draw the event movie as a pixel(y) × frame(x) raster (raw or residual polarities).
fn raster(ui: &mut egui::Ui, frames: &VecDeque<DvsFrame>, residual: bool) {
    let cell = 6.0_f32;
    let size = egui::vec2(DVS_WINDOW as f32 * cell, P_PIXELS as f32 * cell);
    let (rect, _resp) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, RASTER_BG);
    for (fi, frame) in frames.iter().enumerate() {
        let (on, off) = if residual {
            (&frame.on_resid, &frame.off_resid)
        } else {
            (&frame.on_raw, &frame.off_raw)
        };
        let x = rect.left() + fi as f32 * cell;
        for (pi, (&o, &f)) in on.iter().zip(off).enumerate() {
            let color = if o == 1 {
                ON_COLOR
            } else if f == 1 {
                OFF_COLOR
            } else {
                continue;
            };
            let y = rect.top() + pi as f32 * cell;
            painter.rect_filled(
                egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(cell - 1.0, cell - 1.0)),
                1.0,
                color,
            );
        }
    }
}

/// A single-row strip showing where the bar is right now.
fn brightness_strip(ui: &mut egui::Ui, b: &[u8]) {
    let cell = 11.0_f32;
    let size = egui::vec2(b.len() as f32 * cell, cell);
    let (rect, _resp) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    for (i, &v) in b.iter().enumerate() {
        let color = if v == 1 {
            Color32::from_rgb(235, 225, 180)
        } else {
            Color32::from_rgb(32, 34, 40)
        };
        let x = rect.left() + i as f32 * cell;
        painter.rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(x, rect.top()),
                egui::vec2(cell - 1.0, cell - 1.0),
            ),
            1.0,
            color,
        );
    }
}

fn legend(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new("ON = bar arrives")
                .small()
                .strong()
                .color(ON_COLOR),
        );
        ui.add_space(16.0);
        ui.label(
            RichText::new("OFF = bar leaves")
                .small()
                .strong()
                .color(OFF_COLOR),
        );
    });
}
