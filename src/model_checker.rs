use std::{
    env,
    ffi::OsString,
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, TryRecvError},
    },
    time::{Duration, Instant},
};

#[cfg(not(target_arch = "wasm32"))]
use std::process::{Command, Stdio};

use anyhow::{Context as _, Result, anyhow};
use serde::{Deserialize, Serialize};

#[cfg(not(target_arch = "wasm32"))]
use tempfile::TempDir;

/// PRISM engine selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PrismEngine {
    /// MTBDD-based engine (symbolic).
    Mtbdd,
    /// Sparse matrix engine.
    Sparse,
    /// Hybrid engine (default PRISM choice).
    #[default]
    Hybrid,
    /// Explicit state engine.
    Explicit,
    /// Exact (arbitrary precision) engine.
    Exact,
}

impl PrismEngine {
    pub const ALL: [PrismEngine; 5] = [
        PrismEngine::Hybrid,
        PrismEngine::Sparse,
        PrismEngine::Explicit,
        PrismEngine::Mtbdd,
        PrismEngine::Exact,
    ];

    pub fn label(self) -> &'static str {
        match self {
            PrismEngine::Hybrid => "Hybrid (default)",
            PrismEngine::Sparse => "Sparse",
            PrismEngine::Explicit => "Explicit",
            PrismEngine::Mtbdd => "MTBDD",
            PrismEngine::Exact => "Exact",
        }
    }

    pub fn to_arg(self) -> Option<&'static str> {
        match self {
            PrismEngine::Hybrid => None, // Default, no arg needed
            PrismEngine::Sparse => Some("-sparse"),
            PrismEngine::Explicit => Some("-explicit"),
            PrismEngine::Mtbdd => Some("-mtbdd"),
            PrismEngine::Exact => Some("-exact"),
        }
    }
}

/// PRISM automatic heuristic mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PrismHeuristic {
    /// No automatic tuning (default).
    #[default]
    None,
    /// Optimize for speed.
    Speed,
    /// Optimize for memory usage.
    Memory,
}

impl PrismHeuristic {
    pub const ALL: [PrismHeuristic; 3] = [
        PrismHeuristic::None,
        PrismHeuristic::Speed,
        PrismHeuristic::Memory,
    ];

    pub fn label(self) -> &'static str {
        match self {
            PrismHeuristic::None => "None (manual)",
            PrismHeuristic::Speed => "Speed",
            PrismHeuristic::Memory => "Memory",
        }
    }

    pub fn to_arg(self) -> Option<&'static str> {
        match self {
            PrismHeuristic::None => None,
            PrismHeuristic::Speed => Some("-heuristic speed"),
            PrismHeuristic::Memory => Some("-heuristic memory"),
        }
    }
}

/// Structured PRISM engine and solver options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismEngineOptions {
    /// PRISM engine selection.
    pub engine: PrismEngine,
    /// Automatic heuristic mode.
    pub heuristic: PrismHeuristic,
    /// Java maximum heap size (e.g., "1g", "4g").
    pub java_max_mem: String,
    /// Java stack size (e.g., "4m").
    pub java_stack: String,
    /// CUDD maximum memory (e.g., "1g").
    pub cudd_max_mem: String,
    /// Convergence epsilon for iterative methods.
    pub epsilon: Option<f64>,
    /// Maximum iterations for iterative methods.
    pub max_iters: Option<u32>,
}

impl Default for PrismEngineOptions {
    fn default() -> Self {
        Self {
            engine: PrismEngine::default(),
            heuristic: PrismHeuristic::default(),
            java_max_mem: "1g".to_owned(),
            java_stack: "4m".to_owned(),
            cudd_max_mem: "1g".to_owned(),
            epsilon: None,
            max_iters: None,
        }
    }
}

impl PrismEngineOptions {
    /// Convert options to PRISM CLI arguments.
    pub fn to_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        // Engine selection
        if let Some(engine_arg) = self.engine.to_arg() {
            args.push(engine_arg.to_owned());
        }

        // Heuristic mode
        if let Some(heuristic_arg) = self.heuristic.to_arg() {
            // Split "-heuristic speed" into two args
            for part in heuristic_arg.split_whitespace() {
                args.push(part.to_owned());
            }
        }

        // Memory settings - each flag and value must be separate arguments
        args.push("-javamaxmem".to_owned());
        args.push(self.java_max_mem.clone());
        args.push("-javastack".to_owned());
        args.push(self.java_stack.clone());
        args.push("-cuddmaxmem".to_owned());
        args.push(self.cudd_max_mem.clone());

        // Convergence settings - also split into separate args
        if let Some(eps) = self.epsilon {
            args.push("-epsilon".to_owned());
            args.push(eps.to_string());
        }
        if let Some(iters) = self.max_iters {
            args.push("-maxiters".to_owned());
            args.push(iters.to_string());
        }

        args
    }
}

/// Options forwarded to PRISM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismOptions {
    pub timeout_ms: u64,
    /// Structured engine options.
    pub engine_options: PrismEngineOptions,
    /// Additional raw CLI arguments (for advanced use).
    pub extra_args: Vec<String>,
}

impl Default for PrismOptions {
    fn default() -> Self {
        Self {
            timeout_ms: Duration::from_secs(30).as_millis() as u64,
            engine_options: PrismEngineOptions::default(),
            extra_args: vec![],
        }
    }
}

/// Request payload sent to a model checker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismRequest {
    pub model: String,
    pub properties: Vec<String>,
    pub options: PrismOptions,
}

/// Result for a single property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismPropertyResult {
    pub formula: String,
    pub status: String,
    pub probability: Option<f64>,
    pub raw_output: String,
}

pub type PrismResponse = Vec<PrismPropertyResult>;

pub trait ModelChecker: Send + Sync {
    fn name(&self) -> &'static str;
    /// Run model checking on the given request.
    ///
    /// # Errors
    /// Returns an error if model checking fails due to PRISM issues or invalid input.
    fn check(&self, request: PrismRequest) -> Result<PrismResponse>;
}

/// Background job used by the UI to avoid blocking.
pub struct CheckerJob {
    pub started_at: Instant,
    receiver: Receiver<Result<PrismResponse>>,
    stop_requested: Arc<AtomicBool>,
}

impl CheckerJob {
    pub fn new(receiver: Receiver<Result<PrismResponse>>, stop_flag: Arc<AtomicBool>) -> Self {
        Self {
            started_at: Instant::now(),
            receiver,
            stop_requested: stop_flag,
        }
    }

    /// Request the background check to stop.
    pub fn request_stop(&self) {
        self.stop_requested.store(true, Ordering::SeqCst);
    }

    /// Check if stop has been requested.
    pub fn is_stop_requested(&self) -> bool {
        self.stop_requested.load(Ordering::SeqCst)
    }

    pub fn try_recv(&self) -> Option<Result<PrismResponse>> {
        match self.receiver.try_recv() {
            Ok(res) => Some(res),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err(anyhow!("model checker worker dropped"))),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub struct LocalPrism {
    prism_path: PathBuf,
    prism_home: Option<PathBuf>,
}

#[cfg(not(target_arch = "wasm32"))]
impl LocalPrism {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let provided = path.as_ref();

        // If a directory is provided, assume the PRISM bundle root and pick bin/prism.
        let (prism_path, prism_home) = if provided.is_dir() {
            let home = provided.to_path_buf();
            let bin = home.join("bin").join("prism");
            (bin, Some(home))
        } else {
            // If provided path has no directory component (bare command like "prism"),
            // try to resolve it via `which` to get the full path from the shell's PATH.
            let resolved = if provided.components().count() == 1 {
                Self::resolve_from_path(provided).unwrap_or_else(|| provided.to_path_buf())
            } else {
                provided.to_path_buf()
            };

            // If the executable lives in a bin/ folder, PRISM_HOME should be its parent.
            let prism_home = resolved
                .parent()
                .and_then(|p| {
                    if p.file_name().is_some_and(|name| name == "bin") {
                        p.parent()
                    } else {
                        Some(p)
                    }
                })
                .map(|p| p.to_path_buf());
            (resolved, prism_home)
        };

        Self {
            prism_path,
            prism_home,
        }
    }

    /// Resolve a bare command name to its full path using the shell's PATH.
    /// This is needed because GUI apps on macOS don't inherit the user's shell PATH.
    fn resolve_from_path(cmd: &Path) -> Option<PathBuf> {
        // Use login shell to get the user's full PATH environment
        let output = Command::new("/bin/zsh")
            .args(["-l", "-c", &format!("which {}", cmd.display())])
            .output()
            .ok()?;

        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout);
            let path = path_str.trim();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
        None
    }

    #[expect(clippy::unused_self)]
    fn write_inputs(&self, req: &PrismRequest, dir: &TempDir) -> Result<(PathBuf, PathBuf)> {
        let model_path = dir.path().join("model.pm");
        let props_path = dir.path().join("properties.pctl");
        fs::write(&model_path, &req.model).context("failed to write model file")?;
        fs::write(
            &props_path,
            req.properties
                .iter()
                .map(|prop| format!("{prop};\n"))
                .collect::<String>(),
        )
        .context("failed to write properties file")?;
        Ok((model_path, props_path))
    }

    fn parse_output(formula: &str, raw_output: String) -> PrismPropertyResult {
        let lowercase = raw_output.to_ascii_lowercase();

        // Try to extract probability/result value first
        let probability = raw_output.lines().find_map(|line| {
            let line = line.trim_start();
            if let Some(rest) = line.strip_prefix("Result:") {
                rest.split_whitespace()
                    .next()
                    .and_then(|token| token.parse::<f64>().ok())
            } else {
                None
            }
        });

        // Determine status based on result type
        let status = if lowercase.contains("result: true") {
            "satisfied"
        } else if lowercase.contains("result: false") {
            "violated"
        } else if probability.is_some() {
            // Numeric result (probability or reward query)
            "computed"
        } else if lowercase.contains("error") {
            "error"
        } else {
            "unknown"
        }
        .to_owned();

        PrismPropertyResult {
            formula: formula.to_owned(),
            status,
            probability,
            raw_output,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ModelChecker for LocalPrism {
    fn name(&self) -> &'static str {
        "PRISM CLI"
    }

    fn check(&self, request: PrismRequest) -> Result<PrismResponse> {
        // Default implementation without cancellation support
        self.check_cancellable(request, Arc::new(AtomicBool::new(false)))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl LocalPrism {
    /// Run model checking with cancellation support.
    ///
    /// If `stop_flag` is set to `true`, the PRISM process will be killed and
    /// an error will be returned indicating cancellation.
    pub fn check_cancellable(
        &self,
        request: PrismRequest,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<PrismResponse> {
        let tempdir = tempfile::tempdir().context("failed to create workspace for PRISM")?;
        let (model_path, props_path) = self.write_inputs(&request, &tempdir)?;

        let mut results = Vec::with_capacity(request.properties.len());
        let prism_home = self.prism_home.as_deref();
        let use_direct_java = prism_home
            .map(|home| home.join("lib/prism.jar").exists())
            .unwrap_or(false);
        let classpath = prism_home.map(|home| {
            let home_str = home.to_string_lossy();
            format!(
                "{home_str}/lib/prism.jar:{home_str}/classes:{home_str}:{home_str}/lib/pepa.zip:{home_str}/lib/*"
            )
        });

        for formula in &request.properties {
            // Check for cancellation before starting each property
            if stop_flag.load(Ordering::SeqCst) {
                return Err(anyhow!("Model check cancelled"));
            }

            let opts = &request.options.engine_options;
            let mut cmd = if use_direct_java {
                let mut c = Command::new("java");
                if let (Some(home), Some(cp)) = (prism_home, classpath.as_ref()) {
                    c.current_dir(home);
                    c.env("PRISM_HOME", home);
                    c.env("PRISM_DIR", home);

                    let lib_path = home.join("lib");
                    let mut dyld = OsString::from(lib_path.as_os_str());
                    if let Some(existing) = env::var_os("DYLD_LIBRARY_PATH") {
                        dyld.push(":");
                        dyld.push(existing);
                    }
                    c.env("DYLD_LIBRARY_PATH", dyld.clone());
                    c.env("JAVA_LIBRARY_PATH", dyld);
                    c.arg(format!(
                        "-Djava.library.path={}",
                        lib_path.to_string_lossy()
                    ));
                    c.arg("-classpath").arg(cp);
                }
                // Use engine_options for Java memory settings
                c.arg(format!("-Xmx{}", opts.java_max_mem))
                    .arg(format!("-Xss{}", opts.java_stack))
                    .arg("-Djava.awt.headless=true")
                    .arg("prism.PrismCL");
                c
            } else {
                let mut c = Command::new(&self.prism_path);
                if let Some(home) = prism_home {
                    c.current_dir(home);
                    c.env("PRISM_HOME", home);
                    c.env("PRISM_DIR", home);
                }
                c
            };

            // Add model and property files
            cmd.arg(&model_path)
                .arg(&props_path)
                .arg("-pf")
                .arg(formula);

            // Add structured engine options
            cmd.args(opts.to_args());

            // Add any extra user-provided args
            cmd.args(&request.options.extra_args);

            cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

            // Spawn the process instead of blocking on output()
            let mut child = cmd
                .spawn()
                .with_context(|| format!("failed to run PRISM at {}", self.prism_path.display()))?;

            // Poll the child process while checking for cancellation
            let poll_interval = Duration::from_millis(100);
            loop {
                // Check for cancellation
                if stop_flag.load(Ordering::SeqCst) {
                    // Kill the PRISM process
                    let _ = child.kill();
                    let _ = child.wait(); // Reap the zombie process
                    return Err(anyhow!("Model check cancelled"));
                }

                // Check if process has finished
                match child.try_wait() {
                    Ok(Some(status)) => {
                        // Process finished - collect output
                        let mut stdout = String::new();
                        let mut stderr = String::new();
                        if let Some(ref mut out) = child.stdout {
                            let _ = out.read_to_string(&mut stdout);
                        }
                        if let Some(ref mut err) = child.stderr {
                            let _ = err.read_to_string(&mut stderr);
                        }
                        let raw_output = format!("{stdout}{stderr}");

                        if !status.success() {
                            return Err(anyhow!(
                                "PRISM exited with status {}: {}",
                                status,
                                raw_output.trim()
                            ));
                        }

                        results.push(Self::parse_output(formula, raw_output));
                        break;
                    }
                    Ok(None) => {
                        // Process still running, wait a bit
                        std::thread::sleep(poll_interval);
                    }
                    Err(e) => {
                        return Err(anyhow!("Error waiting for PRISM process: {e}"));
                    }
                }
            }
        }

        Ok(results)
    }
}
