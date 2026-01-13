use std::time::Duration;

use crate::{
    app::{DEMO_PRISM_MODEL, Mode, SimTab, TemplateApp},
    learning::{collect_learning_targets, estimate_firing_probabilities, run_learning_iteration},
    snn::{
        graph::{NodeId, NodeKind},
        prism_gen::{PrismGenConfig, generate_pctl_properties, generate_prism_model},
    },
};

const GRID_SPACING: f32 = 32.0;
const NODE_RADIUS: f32 = 18.0;
const NODE_DIAMETER: f32 = NODE_RADIUS * 2.0;
const EDGE_DEFAULT_WEIGHT: u8 = 100;
const ARROW_HEAD_LENGTH: f32 = 10.0;
const ARROW_HEAD_WIDTH: f32 = 8.0;
const SELECT_HALO_ALPHA: u8 = 70;
const SELECT_STROKE_COLOR: egui::Color32 = egui::Color32::from_rgb(0, 140, 255);

pub fn central_view(app: &mut TemplateApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    match app.mode {
        Mode::Design => design_view(app, ui, ctx),
        Mode::Simulate => simulate_view(app, ui, ctx),
        Mode::Verify => verify_view(app, ui, ctx),
    }
}

fn design_view(app: &mut TemplateApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    ui.horizontal(|ui| {
        ui.label("Palette:");
        for kind in NodeKind::palette() {
            if ui.button(kind.label()).clicked() {
                app.design.pending_node_kind = Some(*kind);
                app.push_log(format!("Click the canvas to place a {}", kind.label()));
            }
        }

        ui.separator();

        if ui.button("Delete selected").clicked() {
            if let Some(node_id) = app.design.selected_node.take() {
                app.design.graph.remove_node(node_id);
                app.design.connecting_from = None;
                app.push_log(format!("Deleted node {}", node_id.0));
            }
        }
        if ui.button("Cancel connect").clicked() {
            app.design.connecting_from = None;
        }
    });
    ui.label(&app.design.canvas_note);
    ui.separator();

    let available = ui.available_size();
    let (rect, response) = ui.allocate_at_least(available, egui::Sense::click());
    let painter = ui.painter_at(rect);
    painter.rect_stroke(
        rect,
        8.0,
        egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
        egui::StrokeKind::Outside,
    );

    if app.design.show_grid {
        draw_grid(&painter, rect, GRID_SPACING, ui.visuals());
    }

    let pointer_pos = response.interact_pointer_pos();
    handle_canvas_clicks(app, rect, &response);
    handle_keyboard_shortcuts(app, ui, ctx);

    draw_edges_interactive(app, ui, &painter, rect, pointer_pos);

    draw_nodes(app, ui, &painter, rect);
}

fn draw_grid(painter: &egui::Painter, rect: egui::Rect, spacing: f32, visuals: &egui::Visuals) {
    // Use a subtle grid color that adapts to theme
    let base = visuals.text_color();
    let color = egui::Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 30);
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

fn handle_canvas_clicks(app: &mut TemplateApp, rect: egui::Rect, response: &egui::Response) {
    if let Some(pos) = response.interact_pointer_pos() {
        if response.double_clicked() {
            let kind = app
                .design
                .pending_node_kind
                .take()
                .unwrap_or(NodeKind::Neuron);
            add_node_at(app, rect, pos, kind);
        } else if response.clicked() {
            if let Some(kind) = app.design.pending_node_kind.take() {
                add_node_at(app, rect, pos, kind);
            } else {
                app.design.selected_node = None;
                app.design.selected_edge = None;
                app.design.connecting_from = None;
            }
            app.record_canvas_interaction(pos);
        }
    }
}

fn handle_keyboard_shortcuts(app: &mut TemplateApp, ui: &egui::Ui, ctx: &egui::Context) {
    // Skip global shortcuts if any widget (e.g., text field) has focus
    let any_widget_focused = ctx.memory(|m| m.focused().is_some());
    if any_widget_focused {
        return;
    }

    let delete =
        ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace));
    if delete {
        // First check for selected edge (more specific)
        if let Some(edge_id) = app.design.selected_edge.take() {
            app.design.graph.remove_edge(edge_id);
            app.push_log(format!("Deleted edge {}", edge_id.0));
        } else if let Some(node_id) = app.design.selected_node.take() {
            app.design.graph.remove_node(node_id);
            app.design.connecting_from = None;
            app.push_log(format!("Deleted node {}", node_id.0));
        }
    }

    let escape = ui.input(|i| i.key_pressed(egui::Key::Escape));
    if escape {
        app.design.connecting_from = None;
    }
}

/// Draw edges with interactive clicking support for edge selection.
/// Also positions weight label offset from the edge line for readability.
#[expect(clippy::needless_pass_by_ref_mut)] // ui.interact() requires &mut self
fn draw_edges_interactive(
    app: &mut TemplateApp,
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    rect: egui::Rect,
    pointer_pos: Option<egui::Pos2>,
) {
    let text_color = ui.visuals().text_color();
    // Collect edge data first to avoid borrowing issues
    let edge_data: Vec<_> = app
        .design
        .graph
        .edges
        .iter()
        .filter_map(|edge| {
            let from_pos = app.design.graph.position_of(edge.from)?;
            let to_pos = app.design.graph.position_of(edge.to)?;
            Some((
                edge.id,
                edge.from,
                edge.to,
                edge.weight,
                edge.is_inhibitory,
                from_pos,
                to_pos,
            ))
        })
        .collect();

    let selected_node = app.design.selected_node;
    let selected_edge = app.design.selected_edge;

    for (edge_id, from_node, to_node, weight, is_inhibitory, from_pos, to_pos) in &edge_data {
        let from_screen = graph_to_screen(rect, *from_pos);
        let to_screen = graph_to_screen(rect, *to_pos);

        // Determine edge color based on selection and inhibitory state
        let mut color = if *is_inhibitory {
            egui::Color32::from_rgb(200, 80, 80) // Red-ish for inhibitory
        } else {
            // Use a visible gray that adapts to theme
            egui::Color32::from_rgba_unmultiplied(
                text_color.r(),
                text_color.g(),
                text_color.b(),
                180,
            )
        };

        if selected_edge == Some(*edge_id) {
            color = egui::Color32::from_rgb(255, 180, 0); // Orange for selected edge
        } else if selected_node == Some(*from_node) || selected_node == Some(*to_node) {
            color = egui::Color32::from_rgb(20, 120, 200);
        }

        draw_directed_edge(painter, from_screen, to_screen, color);

        // Calculate perpendicular offset for label position (Feature 3)
        let dir = to_screen - from_screen;
        let len = dir.length();
        if len > f32::EPSILON {
            let unit = dir / len;
            let perp = egui::vec2(-unit.y, unit.x); // Perpendicular vector
            let offset = perp * 14.0; // 14px offset from edge line
            let label_pos = from_screen.lerp(to_screen, 0.5) + offset;

            painter.text(
                label_pos,
                egui::Align2::CENTER_CENTER,
                format!("{weight:+.2}"),
                egui::FontId::proportional(12.0),
                text_color,
            );
        }

        // Create an invisible hit-test rectangle along the edge for clicking
        let mid_point = from_screen.lerp(to_screen, 0.5);
        let hit_size = 20.0; // Hit area size
        let hit_rect = egui::Rect::from_center_size(mid_point, egui::vec2(hit_size, hit_size));
        let edge_response = ui.interact(
            hit_rect,
            ui.id().with(("edge", edge_id.0)),
            egui::Sense::click(),
        );

        if edge_response.clicked() {
            app.design.selected_edge = Some(*edge_id);
            app.design.selected_node = None; // Deselect node when selecting edge
        }

        // Show selection highlight
        if selected_edge == Some(*edge_id) {
            painter.circle_stroke(
                mid_point,
                10.0,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 180, 0)),
            );
        }
    }

    // Draw connection preview line
    if let (Some(from), Some(pointer)) = (app.design.connecting_from, pointer_pos) {
        if let Some(start) = app.design.graph.position_of(from) {
            let start = graph_to_screen(rect, start);
            painter.line_segment(
                [start, pointer],
                egui::Stroke::new(1.5, egui::Color32::LIGHT_BLUE),
            );
        }
    }
}

fn draw_directed_edge(
    painter: &egui::Painter,
    from: egui::Pos2,
    to: egui::Pos2,
    color: egui::Color32,
) {
    let dir = to - from;
    let len = dir.length();
    if len < f32::EPSILON {
        return;
    }
    let unit = dir / len;

    let arrow_tip = to - unit * NODE_RADIUS;
    let arrow_base = arrow_tip - unit * ARROW_HEAD_LENGTH;
    let shaft_start = from + unit * NODE_RADIUS;
    let shaft_end = arrow_base;

    painter.line_segment([shaft_start, shaft_end], egui::Stroke::new(2.0, color));

    let perp = egui::vec2(-unit.y, unit.x);
    let half_width = ARROW_HEAD_WIDTH * 0.5;
    let left = arrow_base + perp * half_width;
    let right = arrow_base - perp * half_width;
    painter.add(egui::Shape::convex_polygon(
        vec![arrow_tip, left, right],
        color,
        egui::Stroke::NONE,
    ));
}

#[expect(clippy::needless_pass_by_ref_mut)] // ui.interact() requires &mut self
fn draw_nodes(app: &mut TemplateApp, ui: &mut egui::Ui, painter: &egui::Painter, rect: egui::Rect) {
    let shift_down = ui.input(|i| i.modifiers.shift);
    let pointer_delta = ui.input(|i| i.pointer.delta());
    let text_color = ui.visuals().text_color();
    let node_ids: Vec<NodeId> = app.design.graph.nodes.iter().map(|n| n.id).collect();

    for node_id in node_ids {
        let Some(node_snapshot) = app.design.graph.node(node_id).cloned() else {
            continue;
        };

        let pos = graph_to_screen(rect, node_snapshot.position);
        let node_rect = egui::Rect::from_center_size(pos, egui::vec2(NODE_DIAMETER, NODE_DIAMETER));
        let response = ui.interact(
            node_rect,
            ui.id().with(("node", node_id.0)),
            egui::Sense::click_and_drag(),
        );

        if response.drag_started() {
            app.design.drag_anchor = Some((node_id, node_snapshot.position));
        }
        if let Some((anchor_id, _)) = app.design.drag_anchor {
            if anchor_id == node_id && response.dragged() {
                if let Some(node) = app.design.graph.node_mut(node_id) {
                    node.position[0] += pointer_delta.x;
                    node.position[1] += pointer_delta.y;
                }
            }
        }
        if let Some((anchor_id, _)) = app.design.drag_anchor {
            let pointer_down = ui.input(|i| i.pointer.primary_down());
            if anchor_id == node_id && !pointer_down {
                if let Some(node) = app.design.graph.node_mut(node_id) {
                    if app.design.snap_to_grid {
                        node.position = snap_to_grid(node.position, GRID_SPACING);
                    }
                }
                app.design.drag_anchor = None;
            }
        }

        if response.clicked() {
            if shift_down {
                match app.design.connecting_from {
                    Some(from) if from != node_id => {
                        if app
                            .design
                            .graph
                            .add_edge(from, node_id, EDGE_DEFAULT_WEIGHT)
                            .is_some()
                        {
                            app.push_log(format!("Connected {} -> {}", from.0, node_id.0));
                        } else {
                            app.push_log("Connection already exists or invalid");
                        }
                        app.design.connecting_from = None;
                    }
                    _ => {
                        app.design.connecting_from = Some(node_id);
                    }
                }
                app.design.selected_node = Some(node_id);
            } else {
                app.design.selected_node = Some(node_id);
                app.design.connecting_from = None;
            }
            if let Some(pos) = response.interact_pointer_pos() {
                app.record_canvas_interaction(pos);
            }
        }

        let fill = node_color(&app.design.graph, node_id, node_snapshot.kind);
        painter.circle_filled(pos, NODE_RADIUS, fill);
        if app.design.selected_node == Some(node_id) || app.design.connecting_from == Some(node_id)
        {
            painter.circle_filled(
                pos,
                NODE_RADIUS + 6.0,
                egui::Color32::from_rgba_unmultiplied(0, 140, 255, SELECT_HALO_ALPHA),
            );
            painter.circle_stroke(
                pos,
                NODE_RADIUS + 3.0,
                egui::Stroke::new(2.5, SELECT_STROKE_COLOR),
            );
        }
        painter.text(
            pos + egui::vec2(0.0, NODE_RADIUS + 6.0),
            egui::Align2::CENTER_TOP,
            node_snapshot.label,
            egui::FontId::proportional(13.0),
            text_color,
        );
    }
}

fn node_color(graph: &crate::snn::graph::SnnGraph, id: NodeId, _kind: NodeKind) -> egui::Color32 {
    // Use topology-based coloring: inputs are roots, outputs are leaves
    if graph.is_input(id) {
        egui::Color32::from_rgb(255, 200, 120) // Orange for inputs
    } else if graph.is_output(id) {
        egui::Color32::from_rgb(200, 200, 255) // Purple for outputs
    } else {
        egui::Color32::from_rgb(120, 180, 255) // Blue for regular neurons
    }
}

fn snap_to_grid(pos: [f32; 2], spacing: f32) -> [f32; 2] {
    [
        (pos[0] / spacing).round() * spacing,
        (pos[1] / spacing).round() * spacing,
    ]
}

fn graph_to_screen(rect: egui::Rect, position: [f32; 2]) -> egui::Pos2 {
    rect.left_top() + egui::vec2(position[0], position[1])
}

fn add_node_at(app: &mut TemplateApp, rect: egui::Rect, pos: egui::Pos2, kind: NodeKind) {
    let mut graph_pos = screen_to_graph(rect, pos);
    if app.design.snap_to_grid {
        graph_pos = snap_to_grid(graph_pos, GRID_SPACING);
    }
    let label = format!("{} {}", kind.label(), app.design.graph.nodes.len() + 1);
    let id = app.design.graph.add_node(label, kind, graph_pos);
    app.design.selected_node = Some(id);
    app.push_log(format!(
        "Added {} at ({:.1}, {:.1})",
        kind.label(),
        graph_pos[0],
        graph_pos[1]
    ));
}

fn screen_to_graph(rect: egui::Rect, pos: egui::Pos2) -> [f32; 2] {
    let local = pos - rect.left_top();
    [local.x, local.y]
}

fn simulate_view(app: &mut TemplateApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    // Poll for simulation progress and results
    poll_simulation_job(app);

    ui.horizontal(|ui| {
        ui.label("Network:");
        ui.strong(app.selection.network.as_deref().unwrap_or("Pick a network"));
        ui.separator();

        let has_job = app.simulate.simulation_job.is_some();
        let can_run = !has_job && app.selection.network.is_some();

        if ui
            .add_enabled(can_run, egui::Button::new("▶ Run"))
            .clicked()
        {
            start_simulation(app);
        }
        if ui
            .add_enabled(has_job, egui::Button::new("⏹ Stop"))
            .clicked()
        {
            if let Some(ref job) = app.simulate.simulation_job {
                job.request_stop();
            }
        }

        ui.separator();

        // Show progress bar
        let progress = app.simulate.progress;
        ui.add(
            egui::ProgressBar::new(progress)
                .desired_width(160.0)
                .text(format!("{:.0}%", progress * 100.0)),
        );

        // Show spike count if available
        if let Some(ref job) = app.simulate.simulation_job {
            if let Some(ref prog) = job.latest_progress {
                ui.label(format!("Spikes: {}", prog.spike_count));
            }
        } else if let Some(ref result) = app.simulate.last_result {
            let total: u32 = result.history.spike_counts.values().sum();
            ui.label(format!("Spikes: {}", total));
        }
    });

    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.label("View:");
        for tab in SimTab::ALL {
            ui.selectable_value(&mut app.simulate.tab, tab, tab.label());
        }
    });
    ui.separator();

    // Request repaint while simulation is running
    if app.simulate.simulation_job.is_some() {
        ctx.request_repaint_after(Duration::from_millis(100));
    }

    match app.simulate.tab {
        SimTab::Timeline => timeline_tab(app, ui),
        SimTab::Raster => raster_tab(app, ui),
        SimTab::Potentials => potentials_tab(app, ui),
        SimTab::Aggregates => aggregates_tab(app, ui),
    }
}

fn poll_simulation_job(app: &mut TemplateApp) {
    if let Some(ref mut job) = app.simulate.simulation_job {
        // Poll for progress
        if let Some(progress) = job.poll_progress() {
            app.simulate.progress = progress.fraction();
            app.simulate.running = true;
        }

        // Check for completion
        if let Some(result) = job.try_get_result() {
            let total_spikes: u32 = result.history.spike_counts.values().sum();
            app.push_log(format!(
                "Simulation complete: {} steps, {} spikes",
                result.total_steps, total_spikes
            ));
            app.simulate.last_result = Some(result);
            app.simulate.simulation_job = None;
            app.simulate.running = false;
            app.simulate.progress = 1.0;
        }
    }
}

fn start_simulation(app: &mut TemplateApp) {
    // Build config from current settings
    let mut config = app.simulate.config.clone();
    config.duration_ms = app.simulate.duration_ms;
    config.record_spikes = app.simulate.record_spikes;
    config.record_potentials = app.simulate.record_membrane;
    // Use model config from the graph for simulation/verification isomorphism
    config.model = app.design.graph.model_config.clone();

    // Start the job
    let job = crate::simulation::start_simulation_job(app.design.graph.clone(), config);
    app.simulate.simulation_job = Some(job);
    app.simulate.progress = 0.0;
    app.simulate.running = true;
    app.push_log(format!(
        "Simulation started: {} ms",
        app.simulate.duration_ms
    ));
}

fn timeline_tab(app: &mut TemplateApp, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.heading("Simulation Configuration");
        ui.add_space(8.0);

        // Duration settings
        ui.group(|ui| {
            ui.label("⏱ Timing");
            ui.add(
                egui::Slider::new(&mut app.simulate.duration_ms, 10.0..=5000.0)
                    .text("Duration (ms)")
                    .logarithmic(true),
            );
            ui.checkbox(&mut app.simulate.record_spikes, "Record spikes");
            ui.checkbox(
                &mut app.simulate.record_membrane,
                "Record membrane potentials (memory intensive)",
            );
        });

        ui.add_space(8.0);

        // Model Configuration
        ui.group(|ui| {
            ui.label("🧠 Model Configuration");
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.label("Threshold levels:");
                ui.add(
                    egui::Slider::new(&mut app.simulate.config.model.threshold_levels, 1..=10)
                        .show_value(true),
                );
                ui.label("(fewer = faster PRISM)");
            });

            ui.horizontal(|ui| {
                ui.checkbox(&mut app.simulate.config.model.enable_arp, "Enable ARP");
                ui.label(format!("({} steps)", app.design.graph.model_config.arp));
            });

            ui.horizontal(|ui| {
                ui.checkbox(&mut app.simulate.config.model.enable_rrp, "Enable RRP");
                ui.label(format!("({} steps)", app.design.graph.model_config.rrp));
            });
        });

        ui.add_space(8.0);

        // Input Patterns are now in Design tab
        ui.group(|ui| {
            ui.label("📊 Input Patterns");
            ui.add_space(4.0);
            ui.label("Configure input generators in the Design tab:");
            ui.label("Select an input neuron → Inspector → Input Generators");
        });

        ui.add_space(8.0);

        // Seed configuration
        ui.group(|ui| {
            ui.label("🎲 Randomness");
            ui.horizontal(|ui| {
                let mut use_seed = app.simulate.config.seed.is_some();
                if ui.checkbox(&mut use_seed, "Fixed seed:").changed() {
                    app.simulate.config.seed = if use_seed { Some(12345) } else { None };
                }
                if let Some(ref mut seed) = app.simulate.config.seed {
                    ui.add(egui::DragValue::new(seed));
                } else {
                    ui.label("(random each run)");
                }
            });
        });
    });
}

fn raster_tab(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.label("Raster plot");
    ui.add_space(6.0);

    if let Some(ref result) = app.simulate.last_result {
        // Draw actual raster plot
        draw_raster_plot(ui, result, &app.design.graph);
    } else {
        ui.label("Run a simulation to see spike raster.");
        placeholder_plot(ui, "spike raster");
    }
}

fn potentials_tab(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.label("Membrane potentials");
    ui.add_space(6.0);

    if let Some(ref result) = app.simulate.last_result {
        if result.history.potentials.is_empty() {
            ui.label("Enable 'Record Membrane' before running to see potential traces.");
        } else {
            draw_potentials_plot(ui, result, &app.design.graph);
        }
    } else {
        ui.label("Run a simulation to see membrane potential traces.");
        placeholder_plot(ui, "membrane potential traces");
    }
}

fn aggregates_tab(app: &mut TemplateApp, ui: &mut egui::Ui) {
    ui.label("Aggregates");
    ui.add_space(6.0);

    if let Some(ref result) = app.simulate.last_result {
        // Show firing rate table
        draw_aggregate_stats(ui, result, &app.design.graph);
    } else {
        ui.label("Run a simulation to see aggregate statistics.");
        placeholder_plot(ui, "population averages");
    }
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

/// Draw a raster plot showing spike events over time.
fn draw_raster_plot(
    ui: &mut egui::Ui,
    result: &crate::simulation::SimulationResult,
    graph: &crate::snn::graph::SnnGraph,
) {
    let desired = egui::vec2(ui.available_width(), 280.0);
    let (rect, _response) = ui.allocate_at_least(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    // Background
    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(25));
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
        egui::StrokeKind::Outside,
    );

    if result.history.spikes.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No spikes recorded",
            egui::FontId::proportional(15.0),
            egui::Color32::LIGHT_GRAY,
        );
        return;
    }

    // Layout parameters
    let margin = 40.0;
    let plot_rect = egui::Rect::from_min_max(
        egui::pos2(rect.left() + margin, rect.top() + 20.0),
        egui::pos2(rect.right() - 20.0, rect.bottom() - 30.0),
    );

    let neuron_ids: Vec<_> = graph.nodes.iter().map(|n| n.id).collect();
    let num_neurons = neuron_ids.len();

    if num_neurons == 0 {
        return;
    }

    let row_height = plot_rect.height() / num_neurons as f32;
    let time_scale = plot_rect.width() / result.total_steps as f32;

    // Draw axis labels
    painter.text(
        egui::pos2(rect.left() + 10.0, plot_rect.center().y),
        egui::Align2::LEFT_CENTER,
        "Neurons",
        egui::FontId::proportional(11.0),
        egui::Color32::GRAY,
    );

    painter.text(
        egui::pos2(plot_rect.center().x, rect.bottom() - 10.0),
        egui::Align2::CENTER_CENTER,
        format!("Time (0-{:.0}ms)", result.total_time_ms()),
        egui::FontId::proportional(11.0),
        egui::Color32::GRAY,
    );

    // Draw grid lines
    for i in 0..=num_neurons {
        let y = plot_rect.top() + i as f32 * row_height;
        painter.hline(
            plot_rect.x_range(),
            y,
            egui::Stroke::new(0.5, egui::Color32::from_gray(50)),
        );
    }

    // Draw spikes
    for spike in &result.history.spikes {
        if let Some(row_idx) = neuron_ids.iter().position(|&id| id == spike.neuron_id) {
            let x = plot_rect.left() + spike.time_step as f32 * time_scale;
            let y_center = plot_rect.top() + (row_idx as f32 + 0.5) * row_height;

            // Color based on neuron type (input=orange, output=purple, other=cyan)
            let color = if graph.is_input(spike.neuron_id) {
                egui::Color32::from_rgb(255, 150, 50)
            } else {
                egui::Color32::from_rgb(100, 200, 255)
            };

            // Draw spike as a small vertical line
            painter.vline(
                x,
                egui::Rangef::new(y_center - row_height * 0.35, y_center + row_height * 0.35),
                egui::Stroke::new(2.0, color),
            );
        }
    }

    // Draw neuron labels
    for (idx, node) in graph.nodes.iter().enumerate() {
        let y = plot_rect.top() + (idx as f32 + 0.5) * row_height;
        painter.text(
            egui::pos2(plot_rect.left() - 5.0, y),
            egui::Align2::RIGHT_CENTER,
            &node.label,
            egui::FontId::proportional(10.0),
            egui::Color32::GRAY,
        );
    }
}

/// Draw membrane potential traces for neurons.
fn draw_potentials_plot(
    ui: &mut egui::Ui,
    result: &crate::simulation::SimulationResult,
    graph: &crate::snn::graph::SnnGraph,
) {
    let desired = egui::vec2(ui.available_width(), 280.0);
    let (rect, _response) = ui.allocate_at_least(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    // Background
    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(25));
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
        egui::StrokeKind::Outside,
    );

    // Layout
    let margin = 50.0;
    let plot_rect = egui::Rect::from_min_max(
        egui::pos2(rect.left() + margin, rect.top() + 20.0),
        egui::pos2(rect.right() - 20.0, rect.bottom() - 30.0),
    );

    let p_min = result.config.potential_range.0 as f32;
    let p_max = result.config.potential_range.1 as f32;
    let p_range = p_max - p_min;

    // Draw trace for each neuron
    let colors = [
        egui::Color32::from_rgb(100, 200, 255),
        egui::Color32::from_rgb(255, 150, 50),
        egui::Color32::from_rgb(150, 255, 100),
        egui::Color32::from_rgb(255, 100, 150),
        egui::Color32::from_rgb(200, 150, 255),
    ];

    let mut color_idx = 0;
    for node in &graph.nodes {
        if let Some(potentials) = result.history.potentials.get(&node.id) {
            if potentials.is_empty() {
                continue;
            }

            let color = colors[color_idx % colors.len()];
            color_idx += 1;

            let time_scale = plot_rect.width() / potentials.len() as f32;

            // Draw line segments
            let points: Vec<egui::Pos2> = potentials
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let x = plot_rect.left() + i as f32 * time_scale;
                    let y_norm = (p as f32 - p_min) / p_range;
                    let y = plot_rect.bottom() - y_norm * plot_rect.height();
                    egui::pos2(x, y)
                })
                .collect();

            for i in 1..points.len() {
                painter.line_segment([points[i - 1], points[i]], egui::Stroke::new(1.5, color));
            }
        }
    }

    // Axis labels
    painter.text(
        egui::pos2(rect.left() + 5.0, plot_rect.center().y),
        egui::Align2::LEFT_CENTER,
        "mV",
        egui::FontId::proportional(11.0),
        egui::Color32::GRAY,
    );

    painter.text(
        egui::pos2(plot_rect.center().x, rect.bottom() - 10.0),
        egui::Align2::CENTER_CENTER,
        format!("Time (0-{:.0}ms)", result.total_time_ms()),
        egui::FontId::proportional(11.0),
        egui::Color32::GRAY,
    );
}

/// Draw aggregate statistics table.
fn draw_aggregate_stats(
    ui: &mut egui::Ui,
    result: &crate::simulation::SimulationResult,
    graph: &crate::snn::graph::SnnGraph,
) {
    let total_time_ms = result.total_time_ms();

    egui::Grid::new("aggregate_stats_grid")
        .striped(true)
        .num_columns(3)
        .show(ui, |ui| {
            // Header
            ui.strong("Neuron");
            ui.strong("Spikes");
            ui.strong("Firing Rate (Hz)");
            ui.end_row();

            // Data rows
            for node in &graph.nodes {
                let spike_count = result
                    .history
                    .spike_counts
                    .get(&node.id)
                    .copied()
                    .unwrap_or(0);
                let firing_rate = result.history.firing_rate(node.id, total_time_ms);

                ui.label(&node.label);
                ui.label(format!("{}", spike_count));
                ui.label(format!("{:.1}", firing_rate));
                ui.end_row();
            }
        });

    ui.add_space(10.0);

    // Summary stats
    let total_spikes: u32 = result.history.spike_counts.values().sum();
    let avg_rate: f64 = result
        .history
        .spike_counts
        .values()
        .map(|&c| (c as f64 / total_time_ms as f64) * 1000.0)
        .sum::<f64>()
        / graph.nodes.len().max(1) as f64;

    ui.horizontal(|ui| {
        ui.label("Total spikes:");
        ui.strong(format!("{}", total_spikes));
        ui.separator();
        ui.label("Avg firing rate:");
        ui.strong(format!("{:.1} Hz", avg_rate));
        ui.separator();
        ui.label("Duration:");
        ui.strong(format!(
            "{:.0} ms ({} steps)",
            total_time_ms, result.total_steps
        ));
    });
}

#[expect(clippy::too_many_lines, clippy::indexing_slicing)]
fn verify_view(app: &mut TemplateApp, ui: &mut egui::Ui, ctx: &egui::Context) {
    app.poll_model_checker();
    app.poll_training_job();
    if app.verify.job.is_some() || app.verify.training_job.is_some() {
        ctx.request_repaint_after(Duration::from_millis(100));
    }

    ui.horizontal(|ui| {
        ui.label("Property set:");
        ui.strong(
            app.selection
                .property
                .as_deref()
                .unwrap_or("Select a property"),
        );
        ui.separator();
        let run_enabled = app.verify.property_enabled && app.verify.job.is_none();
        if ui
            .add_enabled(run_enabled, egui::Button::new("Run checker"))
            .clicked()
        {
            if let Err(err) = app.start_model_check() {
                app.verify.last_error = Some(err.clone());
                app.push_log(format!("Unable to start model check: {err}"));
            }
        }
        // Cancel button - only enabled when a job is running
        if app.verify.job.is_some() {
            if ui.button("⏹ Cancel").clicked() {
                app.cancel_model_check();
            }
        }
        if ui
            .add_enabled(app.verify.job.is_none(), egui::Button::new("Reset result"))
            .clicked()
        {
            app.verify.last_result = None;
            app.verify.last_error = None;
        }
        ui.separator();
        if ui.button("📄 View Model").clicked() {
            app.verify.show_model_window = true;
        }
        if ui.button("📋 View Properties").clicked() {
            app.verify.show_properties_window = true;
        }
    });
    ui.separator();

    // PRISM Model Viewer Window
    let mut show_model_window = app.verify.show_model_window;
    egui::Window::new("PRISM Model Viewer")
        .open(&mut show_model_window)
        .default_size(egui::vec2(600.0, 500.0))
        .resizable(true)
        .scroll([true, true])
        .show(ctx, |ui| {
            let model_text = if app.verify.use_generated_model {
                let config = PrismGenConfig {
                    model: app.design.graph.model_config.clone(),
                    ..PrismGenConfig::default()
                };
                generate_prism_model(&app.design.graph, &config)
            } else {
                DEMO_PRISM_MODEL.to_owned()
            };

            ui.horizontal(|ui| {
                if ui.button("📋 Copy to Clipboard").clicked() {
                    ctx.copy_text(model_text.clone());
                    app.push_log("PRISM model copied to clipboard");
                }
                ui.separator();
                ui.label(format!("{} lines", model_text.lines().count()));
            });
            ui.separator();

            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut model_text.as_str())
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY)
                            .desired_rows(30),
                    );
                });
        });
    app.verify.show_model_window = show_model_window;

    // PCTL Properties Viewer Window
    let mut show_properties_window = app.verify.show_properties_window;
    egui::Window::new("PCTL Properties Viewer")
        .open(&mut show_properties_window)
        .default_size(egui::vec2(550.0, 400.0))
        .resizable(true)
        .scroll([true, true])
        .show(ctx, |ui| {
            // The actual .pctl file content that will be sent to PRISM
            // This matches what write_inputs() does in model_checker.rs
            let actual_pctl_content = format!("{};\n", app.verify.current_formula);

            ui.heading("Properties File (.pctl)");
            ui.label("This is what will actually be written to the .pctl file and sent to PRISM:");
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                if ui.button("📋 Copy to Clipboard").clicked() {
                    ctx.copy_text(actual_pctl_content.clone());
                    app.push_log("PCTL properties copied to clipboard");
                }
            });
            ui.add_space(4.0);

            // Show the actual content in a prominent box
            ui.group(|ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut actual_pctl_content.as_str())
                        .font(egui::TextStyle::Monospace)
                        .desired_width(f32::INFINITY)
                        .desired_rows(3),
                );
            });

            ui.add_space(12.0);
            ui.separator();

            // Collapsible section for generated property templates
            ui.collapsing("📚 Property Templates (for reference)", |ui| {
                ui.label("Auto-generated PCTL property examples you can copy:");
                ui.add_space(4.0);

                let templates = generate_pctl_properties(&app.design.graph);
                egui::ScrollArea::vertical()
                    .max_height(250.0)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut templates.as_str())
                                .font(egui::TextStyle::Monospace)
                                .desired_width(f32::INFINITY)
                                .desired_rows(15),
                        );
                    });
            });
        });
    app.verify.show_properties_window = show_properties_window;

    ui.horizontal(|ui| {
        ui.label("Status:");
        if let Some(job) = app.verify.job.as_ref() {
            ui.spinner();
            if job.is_stop_requested() {
                ui.label(format!("Cancelling… {:.1?}", job.started_at.elapsed()));
            } else {
                ui.label(format!("Running… {:.1?}", job.started_at.elapsed()));
            }
        } else if let Some(err) = &app.verify.last_error {
            ui.colored_label(
                egui::Color32::from_rgb(180, 50, 50),
                format!("Error: {err}"),
            );
        } else if let Some(results) = &app.verify.last_result {
            ui.label(format!(
                "Completed ({} result{})",
                results.len(),
                if results.len() == 1 { "" } else { "s" }
            ));
        } else {
            ui.label("Idle");
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
            columns[1].label("PRISM model preview");
            let model_text = if app.verify.use_generated_model {
                let config = PrismGenConfig {
                    model: app.design.graph.model_config.clone(),
                    ..PrismGenConfig::default()
                };
                generate_prism_model(&app.design.graph, &config)
            } else {
                DEMO_PRISM_MODEL.to_owned()
            };
            egui::ScrollArea::vertical()
                .max_height(300.0)
                .show(&mut columns[1], |ui| {
                    ui.monospace(&model_text);
                });
        }

        columns[1].separator();
        columns[1].label("Results");
        columns[1].add_space(4.0);
        if let Some(results) = &app.verify.last_result {
            for (idx, res) in results.iter().enumerate() {
                columns[1].group(|ui| {
                    ui.strong(format!("Property {} status: {}", idx + 1, res.status));
                    ui.label(format!("Formula: {}", res.formula));
                    if let Some(prob) = res.probability {
                        ui.label(format!("Probability: {prob:.6}"));
                    }
                    ui.collapsing("Raw output", |ui| {
                        egui::ScrollArea::vertical()
                            .max_height(150.0)
                            .show(ui, |ui| {
                                ui.monospace(res.raw_output.trim());
                            });
                    });
                });
            }
        } else if let Some(err) = &app.verify.last_error {
            columns[1].colored_label(egui::Color32::from_rgb(180, 50, 50), err);
            columns[1].label("Try adjusting the PRISM path or formula and rerun.");
        } else {
            columns[1].label("No verification runs yet.");
        }

        // Learning section
        columns[1].separator();
        columns[1].heading("Parameter Learning");
        columns[1].add_space(4.0);

        // Learning config
        columns[1].horizontal(|ui| {
            ui.label("Target prob:");
            ui.add(
                egui::DragValue::new(&mut app.verify.learning_config.target_probability)
                    .speed(0.01)
                    .range(0.0..=1.0),
            );
        });
        columns[1].horizontal(|ui| {
            ui.label("Learning rate:");
            ui.add(
                egui::DragValue::new(&mut app.verify.learning_config.learning_rate)
                    .speed(0.01)
                    .range(0.001..=1.0),
            );
        });
        columns[1].horizontal(|ui| {
            ui.label("Convergence:");
            ui.add(
                egui::DragValue::new(&mut app.verify.learning_config.convergence_threshold)
                    .speed(0.001)
                    .range(0.001..=0.5),
            );
        });
        columns[1].add_space(4.0);
        columns[1].horizontal(|ui| {
            if ui.button("🎲 Randomize Weights").clicked() {
                app.design.graph.randomize_weights_symmetric(100);
                app.push_log("Weights randomized to [0, 100]".to_owned());
            }
        });

        // Run learning button
        let can_learn = app
            .verify
            .last_result
            .as_ref()
            .and_then(|r| r.first())
            .and_then(|r| r.probability)
            .is_some();

        columns[1].add_space(4.0);
        columns[1].horizontal(|ui| {
            if ui
                .add_enabled(can_learn, egui::Button::new("▶ Run Learning Step"))
                .clicked()
            {
                // Get current probability from last result
                if let Some(current_prob) = app
                    .verify
                    .last_result
                    .as_ref()
                    .and_then(|r| r.first())
                    .and_then(|r| r.probability)
                {
                    let targets = collect_learning_targets(&app.design.graph);
                    let firing_probs = estimate_firing_probabilities(&app.design.graph);

                    let result = run_learning_iteration(
                        &mut app.design.graph,
                        &targets,
                        current_prob,
                        &firing_probs,
                        &app.verify.learning_config,
                    );

                    // Update learning state
                    app.verify.learning_state.iteration += 1;
                    app.verify
                        .learning_state
                        .probability_history
                        .push(current_prob);
                    app.verify
                        .learning_state
                        .weight_changes
                        .push(result.weight_changes.clone());
                    app.verify.learning_state.converged = result.converged;
                    app.verify.learning_state.final_error = result.error;

                    let msg = format!(
                        "Learning step {}: error={:.4}, {} weight(s) updated",
                        app.verify.learning_state.iteration,
                        result.error,
                        result.weight_changes.len()
                    );
                    app.push_log(msg);

                    if result.converged {
                        app.push_log("Learning converged!".to_owned());
                    }
                }
            }
            if ui.button("Reset").clicked() {
                app.verify.learning_state.reset();
                app.push_log("Learning state reset".to_owned());
            }
        });

        // Full automated training button
        let training_running = app.verify.training_job.is_some();
        let can_start_training = !training_running && app.verify.job.is_none();

        columns[1].add_space(4.0);
        columns[1].horizontal(|ui| {
            if ui
                .add_enabled(
                    can_start_training,
                    egui::Button::new("🔄 Run Full Training"),
                )
                .clicked()
            {
                if let Err(e) = app.start_training() {
                    app.push_log(format!("Failed to start training: {e}"));
                }
            }

            if training_running {
                if ui.button("⏹ Stop").clicked() {
                    if let Some(job) = &app.verify.training_job {
                        job.request_stop();
                        app.push_log("Stop requested for training...".to_owned());
                    }
                }
                ui.spinner();
                if let Some(job) = &app.verify.training_job {
                    let elapsed = job.started_at.elapsed();
                    if let Some(progress) = &job.latest_progress {
                        ui.label(format!(
                            "Iter {}/{}, prob={:.3}, err={:.3} ({:.1?})",
                            progress.iteration,
                            progress.max_iterations,
                            progress.current_probability,
                            progress.error,
                            elapsed
                        ));
                    } else {
                        ui.label(format!("Starting... ({elapsed:.1?})"));
                    }
                }
            }
        });

        if !can_learn && !training_running {
            columns[1].label("Run verification first to get current probability.");
        }

        // Learning progress
        if app.verify.learning_state.iteration > 0 {
            columns[1].add_space(4.0);
            columns[1].group(|ui| {
                ui.label(format!(
                    "Iterations: {}",
                    app.verify.learning_state.iteration
                ));
                ui.label(format!(
                    "Current error: {:.4}",
                    app.verify.learning_state.final_error
                ));
                if app.verify.learning_state.converged {
                    ui.colored_label(egui::Color32::from_rgb(50, 180, 50), "✓ Converged!");
                }
                if !app.verify.learning_state.probability_history.is_empty() {
                    ui.label("Probability history:");
                    let history: String = app
                        .verify
                        .learning_state
                        .probability_history
                        .iter()
                        .map(|p| format!("{p:.3}"))
                        .collect::<Vec<_>>()
                        .join(" → ");
                    ui.monospace(&history);
                }
            });
        }
    });
}
