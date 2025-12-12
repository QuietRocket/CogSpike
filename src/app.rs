use crate::{
    learning::{self, LearningConfig, LearningState},
    model_checker::{CheckerJob, ModelChecker, PrismOptions, PrismRequest, PrismResponse},
    snn::{
        graph::{NodeId, NodeKind, SnnGraph},
        prism_gen::{generate_prism_model, PrismGenConfig},
    },
    ui::{central, inspector_panel, log_panel, project_explorer, top_bar},
};
use std::{
    sync::{mpsc, Arc},
    thread,
};

#[cfg(not(target_arch = "wasm32"))]
use crate::model_checker::LocalPrism;

pub(crate) const DEMO_PRISM_MODEL: &str = r#"
dtmc

module simple
    s : [0..2] init 0;
    [] s=0 -> 0.5 : (s'=1) + 0.5 : (s'=2);
    [] s=1 -> (s'=1);
    [] s=2 -> (s'=2);
endmodule

label "goal_state" = s=1;
label "error_state" = s=2;
"#;

/// The shallow, UI-first application state with a basic PRISM integration stub.
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
    #[serde(skip)]
    pub(crate) model_checker: Option<Arc<dyn ModelChecker>>,
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
            prism_path: default_prism_path(),
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
    /// If true, generate PRISM model from the design graph; otherwise use demo model.
    pub(crate) use_generated_model: bool,
    #[serde(skip)]
    pub(crate) generated_model_cache: Option<String>,
    #[serde(skip)]
    pub(crate) job: Option<CheckerJob>,
    #[serde(skip)]
    pub(crate) last_result: Option<PrismResponse>,
    #[serde(skip)]
    pub(crate) last_error: Option<String>,
    // Learning state
    #[serde(skip)]
    pub(crate) learning_running: bool,
    #[serde(skip)]
    pub(crate) learning_state: LearningState,
    pub(crate) learning_config: LearningConfig,
}

impl Default for VerifyState {
    fn default() -> Self {
        Self {
            current_formula: "P=? [ F \"output_spike\" ]".to_owned(),
            description: "Probability that output neuron spikes".to_owned(),
            show_model_text: false,
            property_enabled: true,
            use_generated_model: true,
            generated_model_cache: None,
            job: None,
            last_result: None,
            last_error: None,
            learning_running: false,
            learning_state: LearningState::default(),
            learning_config: LearningConfig::default(),
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

fn default_prism_path() -> String {
    if let Ok(path) = std::env::var("PRISM_PATH") {
        return path;
    }
    "prism".to_owned()
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
        let backend = BackendChoice::default();
        let model_checker = Self::build_model_checker(&backend.prism_path);
        Self {
            mode: Mode::Design,
            backend,
            selection: Selection::default(),
            demo: DemoProject::default(),
            design: DesignState::default(),
            simulate: SimulateState::default(),
            verify: VerifyState::default(),
            model_checker,
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
            let mut app: Self = eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
            if app.model_checker.is_none() {
                let prism_path = app.backend.prism_path.clone();
                app.model_checker = Self::build_model_checker(&prism_path);
            }
            app
        } else {
            Default::default()
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn build_model_checker(prism_path: &str) -> Option<Arc<dyn ModelChecker>> {
        Some(Arc::new(LocalPrism::new(prism_path)))
    }

    #[cfg(target_arch = "wasm32")]
    fn build_model_checker(_prism_path: &str) -> Option<Arc<dyn ModelChecker>> {
        None
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

    fn prism_request_from_state(&self) -> PrismRequest {
        let model = if self.verify.use_generated_model {
            // Generate from the design graph
            let config = PrismGenConfig::default();
            generate_prism_model(&self.design.graph, &config)
        } else {
            // Use demo model
            DEMO_PRISM_MODEL.to_owned()
        };

        let formula = self.verify.current_formula.clone();

        PrismRequest {
            model,
            properties: vec![formula],
            options: PrismOptions::default(),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn start_model_check(&mut self) -> Result<(), String> {
        Err("Model checking is unavailable on the web build".to_owned())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn start_model_check(&mut self) -> Result<(), String> {
        if self.verify.job.is_some() {
            return Err("Model checker already running".to_owned());
        }

        self.model_checker = Some(Arc::new(LocalPrism::new(&self.backend.prism_path)));

        let Some(checker) = self.model_checker.clone() else {
            return Err("No model checker backend is configured".to_owned());
        };

        let request = self.prism_request_from_state();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let result = checker.check(request);
            let _ = tx.send(result);
        });
        self.verify.job = Some(CheckerJob::new(rx));
        self.verify.last_error = None;
        self.verify.last_result = None;
        self.push_log("Started PRISM model check");
        Ok(())
    }

    pub(crate) fn poll_model_checker(&mut self) {
        let Some(job) = self.verify.job.as_ref() else {
            return;
        };

        if let Some(result) = job.try_recv() {
            let elapsed = job.started_at.elapsed();
            self.verify.job = None;

            match result {
                Ok(response) => {
                    self.verify.last_result = Some(response);
                    self.verify.last_error = None;
                    self.push_log(format!("PRISM finished in {:.1?}", elapsed));
                }
                Err(err) => {
                    self.verify.last_result = None;
                    self.verify.last_error = Some(err.to_string());
                    self.push_log(format!("PRISM failed after {:.1?}: {err}", elapsed));
                }
            }
        }
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
