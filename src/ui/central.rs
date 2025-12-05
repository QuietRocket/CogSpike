use std::time::Duration;

use crate::{
    app::{Mode, SimTab, TemplateApp},
    snn::graph::{NodeId, NodeKind, SnnGraph},
};

const GRID_SPACING: f32 = 32.0;
const NODE_RADIUS: f32 = 18.0;
const NODE_DIAMETER: f32 = NODE_RADIUS * 2.0;
const EDGE_DEFAULT_WEIGHT: f32 = 1.0;
const ARROW_HEAD_LENGTH: f32 = 10.0;
const ARROW_HEAD_WIDTH: f32 = 8.0;
const SELECT_HALO_ALPHA: u8 = 70;
const SELECT_STROKE_COLOR: egui::Color32 = egui::Color32::from_rgb(0, 140, 255);

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
        for (label, kind) in [
            ("Neuron", NodeKind::Neuron),
            ("Population", NodeKind::Population),
            ("Input", NodeKind::Input),
            ("Output", NodeKind::Output),
        ] {
            if ui.button(label).clicked() {
                app.design.pending_node_kind = Some(kind);
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
        draw_grid(&painter, rect, GRID_SPACING);
    }

    let pointer_pos = response.interact_pointer_pos();
    handle_canvas_clicks(app, rect, &response);
    handle_keyboard_shortcuts(app, ui);

    {
        let graph = &app.design.graph;
        draw_edges(
            graph,
            &painter,
            rect,
            app.design.selected_node,
            app.design.connecting_from,
            pointer_pos,
        );
    }

    draw_nodes(app, ui, &painter, rect);
}

fn draw_grid(painter: &egui::Painter, rect: egui::Rect, spacing: f32) {
    let color = egui::Color32::from_rgba_unmultiplied(220, 220, 220, 80);
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
                app.design.connecting_from = None;
            }
            app.record_canvas_interaction(pos);
        }
    }
}

fn handle_keyboard_shortcuts(app: &mut TemplateApp, ui: &egui::Ui) {
    let delete =
        ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace));
    if delete {
        if let Some(node_id) = app.design.selected_node.take() {
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

fn draw_edges(
    graph: &SnnGraph,
    painter: &egui::Painter,
    rect: egui::Rect,
    selected: Option<NodeId>,
    connecting_from: Option<NodeId>,
    pointer_pos: Option<egui::Pos2>,
) {
    for edge in &graph.edges {
        let Some(from_pos) = graph.position_of(edge.from) else {
            continue;
        };
        let Some(to_pos) = graph.position_of(edge.to) else {
            continue;
        };
        let from_screen = graph_to_screen(rect, from_pos);
        let to_screen = graph_to_screen(rect, to_pos);

        let mut color = egui::Color32::from_gray(60);
        if selected == Some(edge.from) || selected == Some(edge.to) {
            color = egui::Color32::from_rgb(20, 120, 200);
        }
        draw_directed_edge(painter, from_screen, to_screen, color);
        let label_pos = from_screen.lerp(to_screen, 0.5);
        painter.text(
            label_pos,
            egui::Align2::CENTER_CENTER,
            format!("{:+.2}", edge.weight),
            egui::FontId::proportional(12.0),
            egui::Color32::from_gray(30),
        );
    }

    if let (Some(from), Some(pointer)) = (connecting_from, pointer_pos) {
        if let Some(start) = graph.position_of(from) {
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

fn draw_nodes(app: &mut TemplateApp, ui: &mut egui::Ui, painter: &egui::Painter, rect: egui::Rect) {
    let shift_down = ui.input(|i| i.modifiers.shift);
    let pointer_delta = ui.input(|i| i.pointer.delta());
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

        let fill = node_color(node_snapshot.kind);
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
            egui::Color32::from_gray(30),
        );
    }
}

fn node_color(kind: NodeKind) -> egui::Color32 {
    match kind {
        NodeKind::Neuron => egui::Color32::from_rgb(120, 180, 255),
        NodeKind::Population => egui::Color32::from_rgb(180, 255, 120),
        NodeKind::Input => egui::Color32::from_rgb(255, 200, 120),
        NodeKind::Output => egui::Color32::from_rgb(200, 200, 255),
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
