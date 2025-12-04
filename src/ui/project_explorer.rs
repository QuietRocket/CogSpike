use crate::app::TemplateApp;

pub fn project_tree(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.heading("Project explorer");
    ui.add_space(6.0);

    ui.collapsing("Networks", |ui| {
        for name in app.demo.networks.clone() {
            let is_selected = app.selection.network.as_deref() == Some(name.as_str());
            if ui.selectable_label(is_selected, &name).clicked() {
                app.selection.network = Some(name.clone());
                app.push_log(format!("Selected network: {name}"));
            }
        }
        if ui.button("+ New network").clicked() {
            app.push_log("Create network (placeholder)");
        }
    });

    ui.collapsing("Simulations", |ui| {
        for sim in app.demo.simulations.clone() {
            let is_selected = app.selection.simulation.as_deref() == Some(sim.as_str());
            if ui.selectable_label(is_selected, &sim).clicked() {
                app.selection.simulation = Some(sim.clone());
                app.push_log(format!("Selected simulation: {sim}"));
            }
        }
        if ui.button("+ New simulation").clicked() {
            app.push_log("Create simulation (placeholder)");
        }
    });

    ui.collapsing("Properties", |ui| {
        for prop in app.demo.properties.clone() {
            let is_selected = app.selection.property.as_deref() == Some(prop.as_str());
            if ui.selectable_label(is_selected, &prop).clicked() {
                app.selection.property = Some(prop.clone());
                app.push_log(format!("Selected property: {prop}"));
            }
        }
        if ui.button("+ New property").clicked() {
            app.push_log("Create property (placeholder)");
        }
    });

    ui.collapsing("Runs", |ui| {
        for run in app.demo.runs.clone() {
            let is_selected = app.selection.run.as_deref() == Some(run.as_str());
            if ui.selectable_label(is_selected, &run).clicked() {
                app.selection.run = Some(run.clone());
                app.push_log(format!("Selected run: {run}"));
            }
        }
        if ui.button("Clear run history").clicked() {
            app.push_log("Clearing run history (placeholder)");
        }
    });
}
