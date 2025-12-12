use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    sync::mpsc::{Receiver, TryRecvError},
    time::{Duration, Instant},
};

#[cfg(not(target_arch = "wasm32"))]
use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

#[cfg(not(target_arch = "wasm32"))]
use tempfile::TempDir;

/// Options forwarded to PRISM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismOptions {
    pub timeout_ms: u64,
    pub extra_args: Vec<String>,
}

impl Default for PrismOptions {
    fn default() -> Self {
        Self {
            timeout_ms: Duration::from_secs(30).as_millis() as u64,
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
    fn check(&self, request: PrismRequest) -> Result<PrismResponse>;
}

/// Background job used by the UI to avoid blocking.
pub struct CheckerJob {
    pub started_at: Instant,
    receiver: Receiver<Result<PrismResponse>>,
}

impl CheckerJob {
    pub fn new(receiver: Receiver<Result<PrismResponse>>) -> Self {
        Self {
            started_at: Instant::now(),
            receiver,
        }
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
            let prism_path = provided.to_path_buf();
            // If the executable lives in a bin/ folder, PRISM_HOME should be its parent.
            let prism_home = prism_path
                .parent()
                .and_then(|p| {
                    if p.file_name().map_or(false, |name| name == "bin") {
                        p.parent()
                    } else {
                        Some(p)
                    }
                })
                .map(|p| p.to_path_buf());
            (prism_path, prism_home)
        };

        Self {
            prism_path,
            prism_home,
        }
    }

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
        let probability = raw_output
            .lines()
            .find_map(|line| {
                let line = line.trim_start();
                if let Some(rest) = line.strip_prefix("Result:") {
                    rest.trim()
                        .split_whitespace()
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
                "{}/lib/prism.jar:{}/classes:{}:{}/lib/pepa.zip:{}/lib/*",
                home_str, home_str, home_str, home_str, home_str
            )
        });

        for formula in &request.properties {
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
                    c.arg("-Djava.library.path=".to_owned() + lib_path.to_string_lossy().as_ref());
                    c.arg("-classpath").arg(cp);
                }
                c.arg("-Xmx1g")
                    .arg("-Xss4m")
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

            cmd.arg(&model_path)
                .arg(&props_path)
                .arg("-pf")
                .arg(formula)
                .args(&request.options.extra_args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());

            let output = cmd
                .output()
                .with_context(|| format!("failed to run PRISM at {:?}", self.prism_path))?;

            let raw_output = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );

            if !output.status.success() {
                return Err(anyhow!(
                    "PRISM exited with status {}: {}",
                    output.status,
                    raw_output.trim()
                ));
            }

            results.push(Self::parse_output(formula, raw_output));
        }

        Ok(results)
    }
}
