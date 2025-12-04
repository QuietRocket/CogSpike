use std::time::Duration;

use crate::app::{Mode, SimTab, TemplateApp};

pub fn central_view(app: &mut TemplateApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    match app.mode {
        Mode::Design => design_view(app, ui),
        Mode::Simulate => simulate_view(app, ui, ctx),
        Mode::Verify => verify_view(app, ui),
    }
}

fn design_view(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label("Palette:");
        if ui.button("Neuron").clicked() {
            app.push_log("Add neuron (placeholder)");
        }
        if ui.button("Population").clicked() {
            app.push_log("Add population (placeholder)");
        }
        if ui.button("Input").clicked() {
            app.push_log("Add input (placeholder)");
        }
        if ui.button("Output").clicked() {
            app.push_log("Add output (placeholder)");
        }
    });
    ui.separator();

    let available = ui.available_size();
    let (rect, response) = ui.allocate_at_least(available, egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);

    painter.rect_stroke(
        rect,
        8.0,
        egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
        egui::StrokeKind::Outside,
    );
    painter.text(
        rect.center_top() + egui::vec2(0.0, 12.0),
        egui::Align2::CENTER_TOP,
        "Graph canvas placeholder",
        egui::FontId::proportional(16.0),
        egui::Color32::LIGHT_GRAY,
    );

    if app.design.show_grid {
        draw_grid(&painter, rect, 32.0);
    }

    // Simple mock nodes and edges
    let node_positions = [
        rect.left_top() + egui::vec2(80.0, 80.0),
        rect.left_top() + egui::vec2(210.0, 160.0),
        rect.left_top() + egui::vec2(360.0, 120.0),
        rect.left_top() + egui::vec2(500.0, 220.0),
    ];
    painter.line_segment(
        [node_positions[0], node_positions[1]],
        egui::Stroke::new(2.0, egui::Color32::LIGHT_BLUE),
    );
    painter.line_segment(
        [node_positions[1], node_positions[2]],
        egui::Stroke::new(2.0, egui::Color32::LIGHT_GREEN),
    );
    painter.line_segment(
        [node_positions[2], node_positions[3]],
        egui::Stroke::new(2.0, egui::Color32::LIGHT_RED),
    );

    for (idx, pos) in node_positions.iter().enumerate() {
        let color = match idx {
            0 => egui::Color32::from_rgb(120, 180, 255),
            1 => egui::Color32::from_rgb(180, 255, 120),
            2 => egui::Color32::from_rgb(255, 180, 120),
            _ => egui::Color32::from_rgb(200, 200, 255),
        };
        painter.circle_filled(*pos, 18.0, color);
        painter.text(
            *pos + egui::vec2(0.0, 24.0),
            egui::Align2::CENTER_TOP,
            format!("Node {}", idx + 1),
            egui::FontId::proportional(13.0),
            egui::Color32::WHITE,
        );
    }

    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            app.record_canvas_interaction(pos);
            app.push_log(format!("Canvas clicked at ({:.1}, {:.1})", pos.x, pos.y));
        }
    }

    if response.dragged() {
        if let Some(pos) = response.interact_pointer_pos() {
            app.record_canvas_interaction(pos);
        }
    }
}

fn draw_grid(painter: &egui::Painter, rect: egui::Rect, spacing: f32) {
    let color = egui::Color32::from_gray(40);
    let stroke = egui::Stroke::new(1.0, color);
    let mut x = rect.left();
    while x < rect.right() {
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            stroke,
        );
        x += spacing;
    }
    let mut y = rect.top();
    while y < rect.bottom() {
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            stroke,
        );
        y += spacing;
    }
}

fn simulate_view(app: &mut TemplateApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    ui.horizontal(|ui| {
        ui.label("Network:");
        ui.strong(app.selection.network.as_deref().unwrap_or("Pick a network"));
        ui.separator();
        if ui
            .button(if app.simulate.running { "Pause" } else { "Run" })
            .clicked()
        {
            app.simulate.running = !app.simulate.running;
            let status = if app.simulate.running {
                "Running"
            } else {
                "Paused"
            };
            app.push_log(format!("Simulation {status}"));
        }
        if ui.button("Stop").clicked() {
            app.simulate.running = false;
            app.simulate.progress = 0.0;
            app.push_log("Simulation stopped");
        }
        ui.separator();
        ui.add(
            egui::ProgressBar::new(app.simulate.progress)
                .desired_width(160.0)
                .text("progress"),
        );
    });

    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.label("View:");
        for tab in SimTab::ALL {
            ui.selectable_value(&mut app.simulate.tab, tab, tab.label());
        }
    });
    ui.separator();

    if app.simulate.running {
        app.simulate.progress = (app.simulate.progress + 0.005).min(1.0);
        ctx.request_repaint_after(Duration::from_millis(16));
        if (app.simulate.progress - 1.0).abs() < f32::EPSILON {
            app.simulate.running = false;
            app.push_log("Simulation complete (placeholder)");
        }
    }

    match app.simulate.tab {
        SimTab::Timeline => timeline_tab(app, ui),
        SimTab::Raster => raster_tab(ui),
        SimTab::Potentials => potentials_tab(ui),
        SimTab::Aggregates => aggregates_tab(ui),
    }
}

fn timeline_tab(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.label("Timeline");
    ui.add_space(4.0);
    ui.label("Batch progress (placeholder)");
    ui.add(egui::Slider::new(&mut app.simulate.duration_ms, 50.0..=1000.0).text("Duration (ms)"));
    ui.label("Use the inspector on the right to adjust integration step and monitors.");
}

fn raster_tab(ui: &mut egui::Ui) {
    ui.label("Raster plot");
    ui.add_space(6.0);
    ui.label("Would render spikes per neuron over time.");
    placeholder_plot(ui, "spike raster");
}

fn potentials_tab(ui: &mut egui::Ui) {
    ui.label("Membrane potentials");
    ui.add_space(6.0);
    placeholder_plot(ui, "membrane potential traces");
}

fn aggregates_tab(ui: &mut egui::Ui) {
    ui.label("Aggregates");
    ui.add_space(6.0);
    ui.label("Stats such as firing rate, spike histograms, etc.");
    placeholder_plot(ui, "population averages");
}

fn placeholder_plot(ui: &mut egui::Ui, label: &str) {
    let desired = egui::vec2(ui.available_width(), 220.0);
    let (rect, _response) = ui.allocate_at_least(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(25));
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
        egui::StrokeKind::Outside,
    );
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        format!("Placeholder: {label}"),
        egui::FontId::proportional(15.0),
        egui::Color32::LIGHT_GRAY,
    );
}

fn verify_view(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label("Property set:");
        ui.strong(
            app.selection
                .property
                .as_deref()
                .unwrap_or("Select a property"),
        );
        ui.separator();
        if ui.button("Run checker").clicked() {
            app.push_log("Model checking requested (placeholder)");
        }
        if ui.button("Batch run").clicked() {
            app.push_log("Batch verification (placeholder)");
        }
    });
    ui.separator();

    ui.columns(2, |columns| {
        columns[0].label("Properties");
        egui::ScrollArea::vertical().show(&mut columns[0], |ui| {
            for prop in &app.demo.properties {
                let selected = app.selection.property.as_deref() == Some(prop.as_str());
                if ui.selectable_label(selected, prop).clicked() {
                    app.selection.property = Some(prop.clone());
                }
            }
        });

        columns[1].label("Editor & results");
        columns[1].add_space(4.0);
        columns[1].text_edit_multiline(&mut app.verify.current_formula);
        columns[1].text_edit_singleline(&mut app.verify.description);
        columns[1].checkbox(&mut app.verify.property_enabled, "Enable");
        columns[1].checkbox(&mut app.verify.show_model_text, "Show model text");
        if app.verify.show_model_text {
            columns[1].add_space(4.0);
            columns[1].label("PRISM model preview (placeholder)");
            columns[1].monospace("module snn_model\n  // generated content\nendmodule");
        }
    });
}
