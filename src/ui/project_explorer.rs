use crate::app::{DeleteTarget, RenameTarget, TemplateApp};
use crate::snn::graph::SnnGraph;

pub fn project_tree(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.heading("Project Explorer");
    ui.add_space(6.0);

    // ── Networks ────────────────────────────────────────────────────────
    let net_default_open = true;
    egui::CollapsingHeader::new("Networks")
        .default_open(net_default_open)
        .show(ui, |ui| {
            let selected = app.selection.network;
            for i in 0..app.networks.len() {
                // Check if we are renaming this item
                if app.rename_target == Some(RenameTarget::Network(i)) {
                    let resp = ui.text_edit_singleline(&mut app.rename_buffer);
                    if resp.lost_focus() || ui.input(|inp| inp.key_pressed(egui::Key::Enter)) {
                        let new_name = app.rename_buffer.trim().to_owned();
                        if !new_name.is_empty() {
                            app.networks[i].name = new_name;
                        }
                        app.rename_target = None;
                    } else {
                        resp.request_focus();
                    }
                } else {
                    let is_selected = selected == Some(i);
                    let resp = ui.selectable_label(is_selected, &app.networks[i].name);
                    if resp.clicked() {
                        app.select_network(i);
                        app.push_log(format!("Selected network: {}", app.networks[i].name));
                    }
                    // Double-click to rename
                    if resp.double_clicked() {
                        app.rename_buffer = app.networks[i].name.clone();
                        app.rename_target = Some(RenameTarget::Network(i));
                    }
                    // Right-click context menu
                    resp.context_menu(|ui| {
                        if ui.button("Rename").clicked() {
                            app.rename_buffer = app.networks[i].name.clone();
                            app.rename_target = Some(RenameTarget::Network(i));
                            ui.close();
                        }
                        if ui.button("Duplicate").clicked() {
                            let mut cloned = app.networks[i].clone();
                            cloned.name = format!("{} (copy)", cloned.name);
                            app.networks.push(cloned);
                            let new_idx = app.networks.len() - 1;
                            app.select_network(new_idx);
                            app.push_log("Duplicated network");
                            ui.close();
                        }
                        if app.networks.len() > 1 && ui.button("Delete").clicked() {
                            app.pending_delete = Some(DeleteTarget::Network(i));
                            ui.close();
                        }
                    });
                }
            }

            if ui.button("+ New network").clicked() {
                let name = format!("Network {}", app.networks.len() + 1);
                app.sync_graph_to_selection();
                app.networks.push(crate::app::ProjectNetwork {
                    name: name.clone(),
                    graph: SnnGraph::default(),
                });
                let new_idx = app.networks.len() - 1;
                app.select_network(new_idx);
                app.push_log(format!("Created network: {name}"));
            }
        });

    ui.add_space(4.0);

    // ── Simulations ────────────────────────────────────────────────────
    egui::CollapsingHeader::new("Simulations")
        .default_open(true)
        .show(ui, |ui| {
            let selected = app.selection.simulation;
            for i in 0..app.simulations.len() {
                if app.rename_target == Some(RenameTarget::Simulation(i)) {
                    let resp = ui.text_edit_singleline(&mut app.rename_buffer);
                    if resp.lost_focus() || ui.input(|inp| inp.key_pressed(egui::Key::Enter)) {
                        let new_name = app.rename_buffer.trim().to_owned();
                        if !new_name.is_empty() {
                            app.simulations[i].name = new_name;
                        }
                        app.rename_target = None;
                    } else {
                        resp.request_focus();
                    }
                } else {
                    let is_selected = selected == Some(i);
                    let resp = ui.selectable_label(is_selected, &app.simulations[i].name);
                    if resp.clicked() {
                        app.select_simulation(i);
                        app.push_log(format!("Selected simulation: {}", app.simulations[i].name));
                    }
                    if resp.double_clicked() {
                        app.rename_buffer = app.simulations[i].name.clone();
                        app.rename_target = Some(RenameTarget::Simulation(i));
                    }
                    resp.context_menu(|ui| {
                        if ui.button("Rename").clicked() {
                            app.rename_buffer = app.simulations[i].name.clone();
                            app.rename_target = Some(RenameTarget::Simulation(i));
                            ui.close();
                        }
                        if ui.button("Duplicate").clicked() {
                            let mut cloned = app.simulations[i].clone();
                            cloned.name = format!("{} (copy)", cloned.name);
                            app.simulations.push(cloned);
                            let new_idx = app.simulations.len() - 1;
                            app.select_simulation(new_idx);
                            app.push_log("Duplicated simulation");
                            ui.close();
                        }
                        if app.simulations.len() > 1 && ui.button("Delete").clicked() {
                            app.pending_delete = Some(DeleteTarget::Simulation(i));
                            ui.close();
                        }
                    });
                }
            }

            if ui.button("+ New simulation").clicked() {
                let name = format!("Simulation {}", app.simulations.len() + 1);
                app.sync_graph_to_selection(); // save current config first
                app.simulations.push(crate::app::ProjectSimulation {
                    name: name.clone(),
                    config: crate::simulation::SimulationConfig::default(),
                });
                let new_idx = app.simulations.len() - 1;
                app.select_simulation(new_idx);
                app.push_log(format!("Created simulation: {name}"));
            }
        });

    ui.add_space(4.0);

    // ── Properties ─────────────────────────────────────────────────────
    egui::CollapsingHeader::new("Properties")
        .default_open(true)
        .show(ui, |ui| {
            let selected = app.selection.property;
            for i in 0..app.properties.len() {
                if app.rename_target == Some(RenameTarget::Property(i)) {
                    let resp = ui.text_edit_singleline(&mut app.rename_buffer);
                    if resp.lost_focus() || ui.input(|inp| inp.key_pressed(egui::Key::Enter)) {
                        let new_name = app.rename_buffer.trim().to_owned();
                        if !new_name.is_empty() {
                            app.properties[i].name = new_name;
                        }
                        app.rename_target = None;
                    } else {
                        resp.request_focus();
                    }
                } else {
                    let is_selected = selected == Some(i);
                    let resp = ui.selectable_label(is_selected, &app.properties[i].name);
                    if resp.clicked() {
                        app.select_property(i);
                        app.push_log(format!("Selected property: {}", app.properties[i].name));
                    }
                    if resp.double_clicked() {
                        app.rename_buffer = app.properties[i].name.clone();
                        app.rename_target = Some(RenameTarget::Property(i));
                    }
                    resp.context_menu(|ui| {
                        if ui.button("Rename").clicked() {
                            app.rename_buffer = app.properties[i].name.clone();
                            app.rename_target = Some(RenameTarget::Property(i));
                            ui.close();
                        }
                        if ui.button("Duplicate").clicked() {
                            let mut cloned = app.properties[i].clone();
                            cloned.name = format!("{} (copy)", cloned.name);
                            app.properties.push(cloned);
                            let new_idx = app.properties.len() - 1;
                            app.select_property(new_idx);
                            app.push_log("Duplicated property");
                            ui.close();
                        }
                        if app.properties.len() > 1 && ui.button("Delete").clicked() {
                            app.pending_delete = Some(DeleteTarget::Property(i));
                            ui.close();
                        }
                    });
                }
            }

            if ui.button("+ New property").clicked() {
                let name = format!("Property {}", app.properties.len() + 1);
                app.properties.push(crate::app::ProjectProperty {
                    name: name.clone(),
                    description: String::new(),
                    formula: "P=? [ F \"output_spike\" ]".to_owned(),
                });
                let new_idx = app.properties.len() - 1;
                app.select_property(new_idx);
                app.push_log(format!("Created property: {name}"));
            }
        });

    ui.add_space(4.0);

    // ── Simulation Runs ────────────────────────────────────────────────
    egui::CollapsingHeader::new("Simulation Runs")
        .default_open(false)
        .show(ui, |ui| {
            if app.sim_runs.is_empty() {
                ui.label("No simulation runs yet");
            } else {
                let items: Vec<(usize, String, String)> = app
                    .sim_runs
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        let total_spikes: u32 = r.result.history.spike_counts.values().sum();
                        let summary =
                            format!("{} spikes, {} steps", total_spikes, r.result.total_steps);
                        (i, format!("▶ {} — {}", r.label, r.network_name), summary)
                    })
                    .collect();
                let selected = app.selection.sim_run;
                let mut clicked = None;
                for (idx, label, summary) in &items {
                    let is_selected = selected == Some(*idx);
                    let resp = ui.selectable_label(is_selected, label);
                    if resp.clicked() {
                        clicked = Some(*idx);
                    }
                    resp.on_hover_text(summary);
                }
                if let Some(idx) = clicked {
                    app.restore_sim_run(idx);
                }
            }
            if !app.sim_runs.is_empty() && ui.button("Clear").clicked() {
                app.sim_runs.clear();
                app.selection.sim_run = None;
                app.push_log("Cleared simulation runs");
            }
        });

    ui.add_space(4.0);

    // ── Verification Runs ──────────────────────────────────────────────
    egui::CollapsingHeader::new("Verification Runs")
        .default_open(false)
        .show(ui, |ui| {
            if app.verify_runs.is_empty() {
                ui.label("No verification runs yet");
            } else {
                let items: Vec<(usize, String, String)> = app
                    .verify_runs
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        let status = r
                            .result
                            .first()
                            .map(|pr| {
                                if let Some(p) = pr.probability {
                                    format!("{}: {:.6}", pr.status, p)
                                } else {
                                    pr.status.clone()
                                }
                            })
                            .unwrap_or_else(|| "—".to_owned());
                        (i, format!("✓ {} — {}", r.label, r.network_name), status)
                    })
                    .collect();
                let selected = app.selection.verify_run;
                let mut clicked = None;
                for (idx, label, status) in &items {
                    let is_selected = selected == Some(*idx);
                    let resp = ui.selectable_label(is_selected, label);
                    if resp.clicked() {
                        clicked = Some(*idx);
                    }
                    resp.on_hover_text(status);
                }
                if let Some(idx) = clicked {
                    app.restore_verify_run(idx);
                }
            }
            if !app.verify_runs.is_empty() && ui.button("Clear").clicked() {
                app.verify_runs.clear();
                app.selection.verify_run = None;
                app.push_log("Cleared verification runs");
            }
        });

    // ── Delete confirmation modal ──────────────────────────────────────
    if let Some(ref target) = app.pending_delete.clone() {
        let name = match target {
            DeleteTarget::Network(i) => format!("network \"{}\"", app.networks[*i].name),
            DeleteTarget::Simulation(i) => format!("simulation \"{}\"", app.simulations[*i].name),
            DeleteTarget::Property(i) => format!("property \"{}\"", app.properties[*i].name),
        };

        egui::Window::new("Confirm Delete")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ui.ctx(), |ui| {
                ui.label(format!("Delete {name}?"));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        app.pending_delete = None;
                    }
                    if ui.button("Delete").clicked() {
                        match target {
                            DeleteTarget::Network(i) => {
                                app.networks.remove(*i);
                                // Fix selection
                                if app.networks.is_empty() {
                                    app.selection.network = None;
                                } else {
                                    let new_sel = i.saturating_sub(1).min(app.networks.len() - 1);
                                    app.selection.network = Some(new_sel);
                                    app.sync_graph_from_selection();
                                }
                            }
                            DeleteTarget::Simulation(i) => {
                                app.simulations.remove(*i);
                                if app.simulations.is_empty() {
                                    app.selection.simulation = None;
                                } else {
                                    let new_sel =
                                        i.saturating_sub(1).min(app.simulations.len() - 1);
                                    app.selection.simulation = Some(new_sel);
                                }
                            }
                            DeleteTarget::Property(i) => {
                                app.properties.remove(*i);
                                if app.properties.is_empty() {
                                    app.selection.property = None;
                                } else {
                                    let new_sel = i.saturating_sub(1).min(app.properties.len() - 1);
                                    app.selection.property = Some(new_sel);
                                    // Sync the newly selected property's formula
                                    if let Some(prop) = app.properties.get(new_sel) {
                                        app.verify.current_formula = prop.formula.clone();
                                        app.verify.description = prop.description.clone();
                                    }
                                }
                            }
                        }
                        app.pending_delete = None;
                        app.push_log(format!("Deleted {name}"));
                    }
                });
            });
    }
}
