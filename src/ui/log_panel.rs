use crate::app::TemplateApp;

pub fn log_panel(app: &mut TemplateApp, ui: &mut egui::Ui) {
    let spacing = ui.spacing().clone();
    let separator_thickness = ui.style().visuals.widgets.noninteractive.bg_stroke.width;

    // Header
    ui.horizontal(|ui| {
        ui.heading("Logs");
        ui.checkbox(&mut app.follow_logs, "Follow");
        if ui.button("Clear").clicked() {
            app.log_messages.clear();
        }
    });
    ui.separator();

    // Body (scrollable)
    let footer_height = spacing.interact_size.y;
    let scroll_max_height =
        (ui.available_height() - footer_height - separator_thickness - spacing.item_spacing.y)
            .max(0.0);
    let scroll_output = egui::ScrollArea::vertical()
        .id_salt("log_scroll_area")
        .stick_to_bottom(app.follow_logs)
        .auto_shrink([false; 2])
        .max_height(scroll_max_height)
        .show(ui, |ui| {
            for message in &app.log_messages {
                ui.label(message);
            }
        });

    let max_offset = (scroll_output.content_size.y - scroll_output.inner_rect.height()).max(0.0);
    let not_at_bottom = scroll_output.state.offset.y + 1.0 < max_offset;
    if app.follow_logs && not_at_bottom {
        app.follow_logs = false;
    }

    // Footer
    ui.separator();
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(&mut app.draft_log).hint_text("Append a log message"));
        if ui.button("Add").clicked() && !app.draft_log.trim().is_empty() {
            app.push_log(app.draft_log.trim().to_owned());
            app.draft_log.clear();
        }
    });
}
