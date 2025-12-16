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

    if let Some(node_id) = app.design.selected_node {
        ui.separator();
        ui.heading("Selected node");
        if let Some(node) = app.design.graph.node_mut(node_id) {
            ui.label(format!("ID {}", node_id.0));
            ui.horizontal(|ui| {
                ui.label("Label");
                ui.text_edit_singleline(&mut node.label);
            });
            ui.label(format!("Kind: {}", node.kind.label()));

            ui.collapsing("Neuron parameters", |ui| {
                egui::Grid::new("neuron_params_grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("P_rth");
                        ui.add(egui::DragValue::new(&mut node.params.p_rth).speed(0.05));
                        ui.end_row();
                        ui.label("P_rest");
                        ui.add(egui::DragValue::new(&mut node.params.p_rest).speed(0.05));
                        ui.end_row();
                        ui.label("P_reset");
                        ui.add(egui::DragValue::new(&mut node.params.p_reset).speed(0.05));
                        ui.end_row();
                        ui.label("Leak r");
                        ui.add(
                            egui::DragValue::new(&mut node.params.leak_r)
                                .speed(0.01)
                                .range(0.0..=1.0),
                        );
                        ui.end_row();
                        ui.label("ARP");
                        ui.add(
                            egui::DragValue::new(&mut node.params.arp)
                                .speed(1.0)
                                .range(0..=256),
                        );
                        ui.end_row();
                        ui.label("RRP");
                        ui.add(
                            egui::DragValue::new(&mut node.params.rrp)
                                .speed(1.0)
                                .range(0..=256),
                        );
                        ui.end_row();
                        ui.label("alpha");
                        ui.add(
                            egui::DragValue::new(&mut node.params.alpha)
                                .speed(0.01)
                                .range(0.0..=1.0),
                        );
                        ui.end_row();
                        ui.label("dt");
                        ui.add(
                            egui::DragValue::new(&mut node.params.dt)
                                .speed(0.01)
                                .range(0.001..=10.0),
                        );
                        ui.end_row();
                    });
            });

            ui.collapsing("Thresholds (fraction of P_rth)", |ui| {
                egui::Grid::new("thresholds_grid")
                    .num_columns(4)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        for (idx, threshold) in node.params.thresholds.iter_mut().enumerate() {
                            ui.label(format!("th{}", idx + 1));
                            ui.add(egui::DragValue::new(threshold).speed(0.01).range(0.0..=1.2));
                            if idx % 2 == 1 {
                                ui.end_row();
                            }
                        }
                    });
            });

            // Supervisor-specific parameters
            if node.kind == crate::snn::graph::NodeKind::Supervisor {
                ui.separator();
                ui.heading("Supervisor Settings");
                ui.horizontal(|ui| {
                    ui.label("Target Prob");
                    ui.add(
                        egui::DragValue::new(&mut node.supervisor_params.target_probability)
                            .speed(0.01)
                            .range(0.0..=1.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Formula");
                    ui.text_edit_singleline(&mut node.supervisor_params.target_formula);
                });
                ui.label("Connects to the neuron it supervises for learning.");
            }
        } else {
            app.design.selected_node = None;
            ui.label("Selected node no longer exists");
        }
    }

    // Edge inspector section
    if let Some(edge_id) = app.design.selected_edge {
        ui.separator();
        ui.heading("Selected edge");

        // First get edge info immutably to avoid borrow conflicts
        let edge_info = app.design.graph.edge(edge_id).map(|e| {
            let from_node = e.from;
            let to_node = e.to;
            let from_label = app
                .design
                .graph
                .node(from_node)
                .map(|n| n.label.clone())
                .unwrap_or_else(|| format!("Node {}", from_node.0));
            let to_label = app
                .design
                .graph
                .node(to_node)
                .map(|n| n.label.clone())
                .unwrap_or_else(|| format!("Node {}", to_node.0));
            (from_label, to_label)
        });

        if let Some((from_label, to_label)) = edge_info {
            ui.label(format!("Edge {} → {}", from_label, to_label));
            ui.label(format!("ID: {}", edge_id.0));

            // Now borrow mutably for editing
            if let Some(edge) = app.design.graph.edge_mut(edge_id) {
                ui.horizontal(|ui| {
                    ui.label("Weight");
                    ui.add(
                        egui::DragValue::new(&mut edge.weight)
                            .speed(0.01)
                            .range(0.0..=1.0),
                    );
                });

                ui.horizontal(|ui| {
                    ui.checkbox(&mut edge.is_inhibitory, "Inhibitory");
                    if edge.is_inhibitory {
                        ui.label("(suppresses firing)");
                    } else {
                        ui.label("(excitatory)");
                    }
                });
            }

            if ui.button("Deselect edge").clicked() {
                app.design.selected_edge = None;
            }

            if ui.button("🗑 Delete edge").clicked() {
                app.design.graph.remove_edge(edge_id);
                app.design.selected_edge = None;
            }
        } else {
            app.design.selected_edge = None;
            ui.label("Selected edge no longer exists");
        }
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
    ui.label("Model Source");
    ui.checkbox(&mut app.verify.use_generated_model, "Use network graph");
    if !app.verify.use_generated_model {
        ui.label("Using demo DTMC model");
    } else {
        let n_neurons = app
            .design
            .graph
            .nodes
            .iter()
            .filter(|n| n.kind != crate::snn::graph::NodeKind::Input)
            .count();
        ui.label(format!("Generating from {} neurons", n_neurons));
    }

    ui.separator();
    ui.label("Property");
    ui.checkbox(&mut app.verify.property_enabled, "Enabled");
    ui.text_edit_singleline(&mut app.verify.description);
    ui.add_space(4.0);
    ui.label("PCTL Formula:");
    ui.text_edit_multiline(&mut app.verify.current_formula);

    ui.separator();
    ui.checkbox(&mut app.verify.show_model_text, "Show PRISM model");
}
