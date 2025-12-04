use crate::app::{Mode, TemplateApp};

pub fn top_toolbar(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.heading("CogSpike");
        ui.separator();

        ui.menu_button("Project", |ui| {
            if ui.button("New project").clicked() {
                app.push_log("Project > New (placeholder)");
                ui.close();
            }
            if ui.button("Open…").clicked() {
                app.push_log("Project > Open (placeholder)");
                ui.close();
            }
            if ui.button("Save").clicked() {
                app.push_log("Project > Save (placeholder)");
                ui.close();
            }
        });

        ui.separator();
        ui.label("Mode:");
        for mode in Mode::ALL {
            if ui
                .selectable_value(&mut app.mode, mode, mode.label())
                .clicked()
            {
                app.push_log(format!("Switched to {} mode", mode.label()));
            }
        }

        ui.separator();
        if ui.button("Run simulation").clicked() {
            app.mode = Mode::Simulate;
            app.push_log("Requested simulation run");
            app.simulate.running = true;
        }
        if ui.button("Check properties").clicked() {
            app.mode = Mode::Verify;
            app.push_log("Requested property check");
        }
        if ui.button("Logs").clicked() {
            app.log_window_open = true;
        }

        ui.separator();
        egui::ComboBox::from_label("Backend")
            .selected_text(app.backend_label())
            .width(120.0)
            .show_ui(ui, |ui| {
                let options = app.backend.available.clone();
                for (idx, backend) in options.into_iter().enumerate() {
                    if ui
                        .selectable_value(&mut app.backend.active, idx, backend.clone())
                        .clicked()
                    {
                        app.push_log(format!("Backend set to {backend}"));
                    }
                }
            });

        ui.label("PRISM path");
        ui.text_edit_singleline(&mut app.backend.prism_path);
    });
}
