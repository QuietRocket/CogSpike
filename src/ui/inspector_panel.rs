use crate::app::{Mode, TemplateApp};

pub fn inspector(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.heading("Inspector");
    ui.add_space(6.0);

    match app.mode {
        Mode::Design => design_inspector(app, ui),
        Mode::Simulate => simulate_inspector(app, ui),
        Mode::Verify => verify_inspector(app, ui),
    }
}

fn design_inspector(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.label("Network");
    let current = app
        .selection
        .network
        .as_deref()
        .unwrap_or("Select a network");
    ui.strong(current);
    ui.separator();

    ui.checkbox(&mut app.design.show_grid, "Show grid");
    ui.checkbox(&mut app.design.snap_to_grid, "Snap to grid");
    ui.text_edit_singleline(&mut app.design.canvas_note);

    if let Some([x, y]) = app.design.last_interaction {
        ui.label(format!("Last canvas interaction at ({x:.1}, {y:.1})"));
    } else {
        ui.label("No canvas interaction yet");
    }
}

fn simulate_inspector(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.label("Simulation config");
    ui.horizontal(|ui| {
        ui.label("Duration (ms)");
        ui.add(
            egui::DragValue::new(&mut app.simulate.duration_ms)
                .speed(1.0)
                .range(10.0..=10_000.0),
        );
    });
    ui.horizontal(|ui| {
        ui.label("dt (ms)");
        ui.add(
            egui::DragValue::new(&mut app.simulate.time_step_ms)
                .speed(0.01)
                .range(0.01..=10.0),
        );
    });
    ui.checkbox(&mut app.simulate.live_plotting, "Live plotting");

    ui.separator();
    ui.label("Monitors (placeholder)");
    ui.horizontal(|ui| {
        ui.checkbox(&mut app.simulate.record_spikes, "Spikes");
        ui.checkbox(&mut app.simulate.record_membrane, "Membrane V");
    });
}

fn verify_inspector(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.label("Property details");
    ui.checkbox(&mut app.verify.property_enabled, "Enabled in batch");
    ui.text_edit_singleline(&mut app.verify.description);
    ui.add_space(4.0);
    ui.label("Atomic proposition mapping (placeholder)");
    ui.text_edit_multiline(&mut app.verify.current_formula);
    ui.separator();
    ui.checkbox(&mut app.verify.show_model_text, "Preview generated model");
}
