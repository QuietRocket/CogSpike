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

#[expect(clippy::too_many_lines)]
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

    // Model complexity settings (affects both simulation and verification)
    ui.collapsing("Model Settings", |ui| {
        let old_levels = app.design.graph.model_config.threshold_levels;

        ui.horizontal(|ui| {
            ui.label("Threshold levels");
            ui.add(egui::Slider::new(
                &mut app.design.graph.model_config.threshold_levels,
                1..=10,
            ));
        });

        // When threshold levels change, rescale all neuron thresholds to be evenly distributed
        let new_levels = app.design.graph.model_config.threshold_levels;
        if new_levels != old_levels {
            for node in &mut app.design.graph.nodes {
                for i in 0..10 {
                    // Evenly distribute thresholds: th[i] = (i+1) * 100 / levels
                    // For i < levels, otherwise keep at 100
                    if i < new_levels as usize {
                        node.params.thresholds[i] = (((i + 1) * 100) / new_levels as usize) as u8;
                    } else {
                        node.params.thresholds[i] = 100;
                    }
                }
            }
        }

        ui.checkbox(
            &mut app.design.graph.model_config.enable_arp,
            "Enable ARP (Absolute Refractory)",
        );

        // RRP requires ARP to be enabled
        ui.add_enabled_ui(app.design.graph.model_config.enable_arp, |ui| {
            ui.checkbox(
                &mut app.design.graph.model_config.enable_rrp,
                "Enable RRP (Relative Refractory)",
            );
        });

        // Force RRP off if ARP is disabled
        if !app.design.graph.model_config.enable_arp {
            app.design.graph.model_config.enable_rrp = false;
        }

        ui.separator();
        ui.label("Global Neuron Parameters");

        egui::Grid::new("global_neuron_params")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("P_rth (resting threshold)");
                ui.add(
                    egui::DragValue::new(&mut app.design.graph.model_config.p_rth)
                        .speed(1)
                        .range(1..=100)
                        .suffix("%"),
                );
                ui.end_row();

                ui.label("P_rest (resting potential)");
                ui.add(
                    egui::DragValue::new(&mut app.design.graph.model_config.p_rest)
                        .speed(1)
                        .range(0..=100)
                        .suffix("%"),
                );
                ui.end_row();

                ui.label("P_reset (after spike)");
                ui.add(
                    egui::DragValue::new(&mut app.design.graph.model_config.p_reset)
                        .speed(1)
                        .range(0..=100)
                        .suffix("%"),
                );
                ui.end_row();

                ui.label("Leak rate");
                ui.add(
                    egui::DragValue::new(&mut app.design.graph.model_config.leak_r)
                        .speed(1)
                        .range(0..=100)
                        .suffix("%"),
                );
                ui.end_row();

                // ARP duration (only if ARP enabled)
                if app.design.graph.model_config.enable_arp {
                    ui.label("ARP duration");
                    ui.add(
                        egui::DragValue::new(&mut app.design.graph.model_config.arp)
                            .speed(1)
                            .range(1..=256)
                            .suffix(" steps"),
                    );
                    ui.end_row();
                }

                // RRP duration and alpha (only if RRP enabled)
                if app.design.graph.model_config.enable_rrp {
                    ui.label("RRP duration");
                    ui.add(
                        egui::DragValue::new(&mut app.design.graph.model_config.rrp)
                            .speed(1)
                            .range(1..=256)
                            .suffix(" steps"),
                    );
                    ui.end_row();

                    ui.label("Alpha (RRP firing scale)");
                    ui.add(
                        egui::DragValue::new(&mut app.design.graph.model_config.alpha)
                            .speed(1)
                            .range(0..=100)
                            .suffix("%"),
                    );
                    ui.end_row();
                }
            });
    });

    if let Some(node_id) = app.design.selected_node {
        ui.separator();
        ui.heading("Selected node");

        // Check topology and model config before mutable borrow
        let is_output_node = app.design.graph.is_output(node_id);
        let is_input_node = app.design.graph.is_input(node_id);
        let active_threshold_levels = app.design.graph.model_config.threshold_levels as usize;

        if let Some(node) = app.design.graph.node_mut(node_id) {
            ui.label(format!("ID {}", node_id.0));
            ui.horizontal(|ui| {
                ui.label("Label");
                ui.text_edit_singleline(&mut node.label);
            });
            ui.label(format!("Kind: {}", node.kind.label()));

            // Show per-neuron settings (only dt and thresholds now)
            ui.collapsing("Per-Neuron Settings", |ui| {
                egui::Grid::new("neuron_params_grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("dt (tenths)");
                        ui.add(
                            egui::DragValue::new(&mut node.params.dt)
                                .speed(1)
                                .range(1..=100),
                        );
                        ui.end_row();
                    });
            });

            ui.collapsing("Thresholds (% of P_rth)", |ui| {
                ui.label(format!(
                    "Showing {} of 10 levels (set in Model Settings)",
                    active_threshold_levels
                ));
                egui::Grid::new("thresholds_grid")
                    .num_columns(4)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        for (idx, threshold) in node
                            .params
                            .thresholds
                            .iter_mut()
                            .take(active_threshold_levels)
                            .enumerate()
                        {
                            ui.label(format!("th{}", idx + 1));
                            ui.add(
                                egui::DragValue::new(threshold)
                                    .speed(1)
                                    .range(0..=120)
                                    .suffix("%"),
                            );
                            if idx % 2 == 1 {
                                ui.end_row();
                            }
                        }
                    });
            });

            // Learning target settings (for output nodes - nodes with no outgoing edges)
            if is_output_node {
                ui.separator();
                ui.heading("Learning Target");

                let mut is_target = node.target_probability.is_some();
                if ui
                    .checkbox(&mut is_target, "Enable as learning target")
                    .changed()
                {
                    if is_target {
                        node.target_probability = Some(0.8);
                        node.target_formula = Some("F \"spike\"".to_owned());
                    } else {
                        node.target_probability = None;
                        node.target_formula = None;
                    }
                }

                if let Some(ref mut prob) = node.target_probability {
                    ui.horizontal(|ui| {
                        ui.label("Target Prob");
                        ui.add(egui::DragValue::new(prob).speed(0.01).range(0.0..=1.0));
                    });
                }
                if let Some(ref mut formula) = node.target_formula {
                    ui.horizontal(|ui| {
                        ui.label("Formula");
                        ui.text_edit_singleline(formula);
                    });
                }
            }

            // Input generator settings (for input nodes - nodes with no incoming edges)
            if is_input_node {
                ui.separator();
                ui.heading("Input Generators");

                // Ensure input_config exists
                if node.input_config.is_none() {
                    node.input_config =
                        Some(crate::simulation::InputNeuronConfig::with_default_generator());
                }

                if let Some(ref mut config) = node.input_config {
                    // Combine mode selector
                    ui.horizontal(|ui| {
                        ui.label("Combine:");
                        egui::ComboBox::from_id_salt(format!("combine_mode_{}", node.id.0))
                            .selected_text(config.combine_mode.label())
                            .show_ui(ui, |ui| {
                                for mode in crate::simulation::GeneratorCombineMode::ALL {
                                    ui.selectable_value(
                                        &mut config.combine_mode,
                                        mode,
                                        mode.label(),
                                    );
                                }
                            });
                    });

                    ui.add_space(4.0);

                    // List generators
                    let mut to_remove = None;
                    for (idx, generator) in config.generators.iter_mut().enumerate() {
                        ui.group(|ui| {
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut generator.active, "");
                                ui.text_edit_singleline(&mut generator.label);
                                if ui.small_button("🗑").clicked() {
                                    to_remove = Some(generator.id);
                                }
                            });

                            // Pattern selector
                            ui.horizontal(|ui| {
                                ui.label("Pattern:");
                                egui::ComboBox::from_id_salt(format!(
                                    "pattern_{}_{}",
                                    node.id.0, idx
                                ))
                                .selected_text(generator.pattern.label())
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(
                                            matches!(
                                                generator.pattern,
                                                crate::simulation::InputPattern::AlwaysOn
                                            ),
                                            "Always On",
                                        )
                                        .clicked()
                                    {
                                        generator.pattern =
                                            crate::simulation::InputPattern::AlwaysOn;
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(
                                                generator.pattern,
                                                crate::simulation::InputPattern::AlwaysOff
                                            ),
                                            "Always Off",
                                        )
                                        .clicked()
                                    {
                                        generator.pattern =
                                            crate::simulation::InputPattern::AlwaysOff;
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(
                                                generator.pattern,
                                                crate::simulation::InputPattern::Random { .. }
                                            ),
                                            "Random",
                                        )
                                        .clicked()
                                    {
                                        generator.pattern =
                                            crate::simulation::InputPattern::Random {
                                                probability: 0.5,
                                            };
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(
                                                generator.pattern,
                                                crate::simulation::InputPattern::Periodic { .. }
                                            ),
                                            "Periodic",
                                        )
                                        .clicked()
                                    {
                                        generator.pattern =
                                            crate::simulation::InputPattern::Periodic {
                                                period: 10,
                                                phase: 0,
                                            };
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(
                                                generator.pattern,
                                                crate::simulation::InputPattern::Burst { .. }
                                            ),
                                            "Burst",
                                        )
                                        .clicked()
                                    {
                                        generator.pattern =
                                            crate::simulation::InputPattern::Burst {
                                                burst_length: 3,
                                                silence_length: 5,
                                            };
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(
                                                generator.pattern,
                                                crate::simulation::InputPattern::Pulse { .. }
                                            ),
                                            "Pulse",
                                        )
                                        .clicked()
                                    {
                                        generator.pattern =
                                            crate::simulation::InputPattern::Pulse { duration: 50 };
                                    }
                                    if ui
                                        .selectable_label(
                                            matches!(
                                                generator.pattern,
                                                crate::simulation::InputPattern::Poisson { .. }
                                            ),
                                            "Poisson",
                                        )
                                        .clicked()
                                    {
                                        generator.pattern =
                                            crate::simulation::InputPattern::Poisson {
                                                rate_hz: 100.0,
                                            };
                                    }
                                });
                            });

                            // Pattern-specific parameters
                            match &mut generator.pattern {
                                crate::simulation::InputPattern::Random { probability } => {
                                    ui.add(egui::Slider::new(probability, 0.0..=1.0).text("p"));
                                }
                                crate::simulation::InputPattern::Periodic { period, phase } => {
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::DragValue::new(period)
                                                .prefix("T=")
                                                .range(1..=1000),
                                        );
                                        ui.add(
                                            egui::DragValue::new(phase).prefix("φ=").range(0..=100),
                                        );
                                    });
                                }
                                crate::simulation::InputPattern::Burst {
                                    burst_length,
                                    silence_length,
                                } => {
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::DragValue::new(burst_length)
                                                .prefix("on=")
                                                .range(1..=100),
                                        );
                                        ui.add(
                                            egui::DragValue::new(silence_length)
                                                .prefix("off=")
                                                .range(1..=100),
                                        );
                                    });
                                }
                                crate::simulation::InputPattern::Pulse { duration }
                                | crate::simulation::InputPattern::Silence { duration } => {
                                    ui.add(
                                        egui::DragValue::new(duration)
                                            .prefix("n=")
                                            .range(1..=10000),
                                    );
                                }
                                crate::simulation::InputPattern::Poisson { rate_hz } => {
                                    ui.add(
                                        egui::Slider::new(rate_hz, 1.0..=1000.0)
                                            .text("Hz")
                                            .logarithmic(true),
                                    );
                                }
                                _ => {}
                            }
                        });
                    }

                    // Remove generator if requested
                    if let Some(id) = to_remove {
                        config.remove_generator(id);
                    }

                    // Add generator button
                    if ui.button("➕ Add Generator").clicked() {
                        let gen_num = config.generators.len() + 1;
                        config.add_generator(
                            format!("Gen {}", gen_num),
                            crate::simulation::InputPattern::AlwaysOn,
                        );
                    }
                }
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
            ui.label(format!("Edge {from_label} → {to_label}"));
            ui.label(format!("ID: {}", edge_id.0));

            // Now borrow mutably for editing
            if let Some(edge) = app.design.graph.edge_mut(edge_id) {
                ui.horizontal(|ui| {
                    ui.label("Weight");
                    ui.add(
                        egui::DragValue::new(&mut edge.weight)
                            .speed(1)
                            .range(0..=100)
                            .suffix("%"),
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
            .filter(|n| !app.design.graph.is_input(n.id))
            .count();
        ui.label(format!("Generating from {n_neurons} neurons"));
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

    // PRISM Engine Configuration
    ui.separator();
    ui.collapsing("PRISM Engine Options", |ui| {
        let opts = &mut app.verify.prism_options;

        // Engine selection
        ui.horizontal(|ui| {
            ui.label("Engine:");
            egui::ComboBox::from_id_salt("prism_engine")
                .selected_text(opts.engine.label())
                .show_ui(ui, |ui| {
                    for engine in crate::model_checker::PrismEngine::ALL {
                        ui.selectable_value(&mut opts.engine, engine, engine.label());
                    }
                });
        });

        // Heuristic mode
        ui.horizontal(|ui| {
            ui.label("Heuristic:");
            egui::ComboBox::from_id_salt("prism_heuristic")
                .selected_text(opts.heuristic.label())
                .show_ui(ui, |ui| {
                    for heuristic in crate::model_checker::PrismHeuristic::ALL {
                        ui.selectable_value(&mut opts.heuristic, heuristic, heuristic.label());
                    }
                });
        });

        ui.add_space(4.0);
        ui.label("Memory Settings");

        egui::Grid::new("prism_memory_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Java heap:");
                ui.text_edit_singleline(&mut opts.java_max_mem);
                ui.end_row();

                ui.label("Java stack:");
                ui.text_edit_singleline(&mut opts.java_stack);
                ui.end_row();

                ui.label("CUDD memory:");
                ui.text_edit_singleline(&mut opts.cudd_max_mem);
                ui.end_row();
            });

        ui.add_space(4.0);
        ui.label("Convergence (optional)");

        // Epsilon
        let mut has_epsilon = opts.epsilon.is_some();
        ui.horizontal(|ui| {
            ui.checkbox(&mut has_epsilon, "Epsilon:");
            if has_epsilon {
                let eps = opts.epsilon.get_or_insert(1e-6);
                ui.add(egui::DragValue::new(eps).speed(1e-7).range(1e-12..=0.1));
            } else {
                opts.epsilon = None;
                ui.label("(default)");
            }
        });

        // Max iterations
        let mut has_max_iters = opts.max_iters.is_some();
        ui.horizontal(|ui| {
            ui.checkbox(&mut has_max_iters, "Max iters:");
            if has_max_iters {
                let iters = opts.max_iters.get_or_insert(10000);
                ui.add(
                    egui::DragValue::new(iters)
                        .speed(100)
                        .range(100..=1_000_000),
                );
            } else {
                opts.max_iters = None;
                ui.label("(default)");
            }
        });

        // Reset button
        ui.add_space(4.0);
        if ui.button("Reset to defaults").clicked() {
            *opts = crate::model_checker::PrismEngineOptions::default();
        }
    });
}
