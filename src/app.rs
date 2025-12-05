use crate::{
    snn::graph::{NodeId, NodeKind, SnnGraph},
    ui::{central, inspector_panel, log_panel, project_explorer, top_bar},
};

/// The shallow, UI-first application state. No business logic or backend calls yet.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct TemplateApp {
    pub(crate) mode: Mode,
    pub(crate) backend: BackendChoice,
    pub(crate) selection: Selection,
    pub(crate) demo: DemoProject,
    pub(crate) design: DesignState,
    pub(crate) simulate: SimulateState,
    pub(crate) verify: VerifyState,
    pub(crate) log_messages: Vec<String>,
    pub(crate) follow_logs: bool,
    pub(crate) log_window_open: bool,
    #[serde(skip)]
    pub(crate) draft_log: String,
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Design,
    Simulate,
    Verify,
}

impl Mode {
    pub const ALL: [Mode; 3] = [Mode::Design, Mode::Simulate, Mode::Verify];

    pub fn label(self) -> &'static str {
        match self {
            Mode::Design => "Design",
            Mode::Simulate => "Simulate",
            Mode::Verify => "Verify",
        }
    }
}

impl Default for Mode {
    fn default() -> Self {
        Mode::Design
    }
}

#[derive(serde::Deserialize, serde::Serialize, Default)]
pub struct Selection {
    pub(crate) network: Option<String>,
    pub(crate) simulation: Option<String>,
    pub(crate) property: Option<String>,
    pub(crate) run: Option<String>,
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct BackendChoice {
    pub(crate) available: Vec<String>,
    pub(crate) active: usize,
    pub(crate) prism_path: String,
}

impl Default for BackendChoice {
    fn default() -> Self {
        Self {
            available: vec!["CPU".to_owned(), "GPU".to_owned(), "PRISM CLI".to_owned()],
            active: 0,
            prism_path: "/usr/bin/prism".to_owned(),
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct DesignState {
    pub(crate) show_grid: bool,
    pub(crate) snap_to_grid: bool,
    pub(crate) canvas_note: String,
    pub(crate) last_interaction: Option<[f32; 2]>,
    pub(crate) graph: SnnGraph,
    pub(crate) selected_node: Option<NodeId>,
    pub(crate) connecting_from: Option<NodeId>,
    #[serde(skip)]
    pub(crate) pending_node_kind: Option<NodeKind>,
    #[serde(skip)]
    pub(crate) drag_anchor: Option<(NodeId, [f32; 2])>,
}

impl Default for DesignState {
    fn default() -> Self {
        Self {
            show_grid: true,
            snap_to_grid: true,
            canvas_note: "Double-click to add neurons; Shift-click to connect".to_owned(),
            last_interaction: None,
            graph: SnnGraph::demo_layout(),
            selected_node: None,
            connecting_from: None,
            pending_node_kind: None,
            drag_anchor: None,
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Copy, PartialEq, Eq)]
pub enum SimTab {
    Timeline,
    Raster,
    Potentials,
    Aggregates,
}

impl SimTab {
    pub const ALL: [SimTab; 4] = [
        SimTab::Timeline,
        SimTab::Raster,
        SimTab::Potentials,
        SimTab::Aggregates,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SimTab::Timeline => "Timeline",
            SimTab::Raster => "Raster plot",
            SimTab::Potentials => "Membrane potentials",
            SimTab::Aggregates => "Aggregates",
        }
    }
}

impl Default for SimTab {
    fn default() -> Self {
        SimTab::Timeline
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct SimulateState {
    pub(crate) running: bool,
    pub(crate) progress: f32,
    pub(crate) tab: SimTab,
    pub(crate) selected_run: Option<String>,
    pub(crate) duration_ms: f32,
    pub(crate) time_step_ms: f32,
    pub(crate) live_plotting: bool,
    pub(crate) record_spikes: bool,
    pub(crate) record_membrane: bool,
}

impl Default for SimulateState {
    fn default() -> Self {
        Self {
            running: false,
            progress: 0.0,
            tab: SimTab::Timeline,
            selected_run: None,
            duration_ms: 250.0,
            time_step_ms: 0.1,
            live_plotting: true,
            record_spikes: true,
            record_membrane: true,
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct VerifyState {
    pub(crate) current_formula: String,
    pub(crate) description: String,
    pub(crate) show_model_text: bool,
    pub(crate) property_enabled: bool,
}

impl Default for VerifyState {
    fn default() -> Self {
        Self {
            current_formula: "P>=0.95 [ F<=100 error_state ]".to_owned(),
            description: "Goal reached with high probability within 100 ms".to_owned(),
            show_model_text: false,
            property_enabled: true,
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct DemoProject {
    pub(crate) networks: Vec<String>,
    pub(crate) simulations: Vec<String>,
    pub(crate) properties: Vec<String>,
    pub(crate) runs: Vec<String>,
}

impl Default for DemoProject {
    fn default() -> Self {
        Self {
            networks: vec![
                "Navigation SNN".to_owned(),
                "Obstacle Avoider".to_owned(),
                "Toy Circuit".to_owned(),
            ],
            simulations: vec![
                "Baseline".to_owned(),
                "High noise".to_owned(),
                "Ablation study".to_owned(),
            ],
            properties: vec![
                "Safety: avoid_error".to_owned(),
                "Reachability: goal".to_owned(),
                "Liveness: keep_firing".to_owned(),
            ],
            runs: vec![
                "Sim run #12".to_owned(),
                "Sim run #13".to_owned(),
                "Verify run #4".to_owned(),
            ],
        }
    }
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            mode: Mode::Design,
            backend: BackendChoice::default(),
            selection: Selection::default(),
            demo: DemoProject::default(),
            design: DesignState::default(),
            simulate: SimulateState::default(),
            verify: VerifyState::default(),
            log_messages: vec![
                "Welcome to CogSpike.".to_owned(),
                "Tip: switch modes with the top toolbar.".to_owned(),
                "Log output will appear here.".to_owned(),
            ],
            follow_logs: true,
            log_window_open: true,
            draft_log: String::new(),
        }
    }
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        }
    }

    pub(crate) fn backend_label(&self) -> &str {
        self.backend
            .available
            .get(self.backend.active)
            .map(String::as_str)
            .unwrap_or("Unknown")
    }

    pub(crate) fn push_log(&mut self, message: impl Into<String>) {
        self.log_messages.push(message.into());
    }

    pub(crate) fn record_canvas_interaction(&mut self, pos: egui::Pos2) {
        self.design.last_interaction = Some([pos.x, pos.y]);
    }
}

impl eframe::App for TemplateApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            top_bar::top_toolbar(self, ui);
        });

        egui::SidePanel::left("project_explorer")
            .default_width(240.0)
            .resizable(true)
            .show(ctx, |ui| {
                project_explorer::project_tree(self, ui);
            });

        egui::SidePanel::right("inspector")
            .default_width(280.0)
            .resizable(true)
            .show(ctx, |ui| {
                inspector_panel::inspector(self, ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            central::central_view(self, ui, ctx);
        });

        let mut log_window_open = self.log_window_open;
        egui::Window::new("Logs")
            .open(&mut log_window_open)
            .default_size(egui::vec2(520.0, 240.0))
            .vscroll(false)
            .show(ctx, |ui| {
                log_panel::log_panel(self, ui);
            });
        self.log_window_open = log_window_open;
    }
}
