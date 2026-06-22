//! CLI export tool for CogSpike simulation diagrams.
//!
//! Generates raster plots and membrane potential traces as PNG images
//! for research paper inclusion.
//!
//! **Note:** This tool uses the paper's _additive_ leak model
//! `p_{t+1} = max(0, p_t + λ_d + input)` rather than the simulation engine's
//! multiplicative model `p_{t+1} = floor(r * p_t + input)`. This ensures
//! the generated diagrams match the theoretical predictions in the paper.

use clap::Parser;
use plotters::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::path::PathBuf;

/// CogSpike diagram export tool.
#[derive(Parser)]
#[command(name = "export", about = "Generate simulation diagram PNGs")]
struct Cli {
    /// Which scenario(s) to run: 1, 2, 3, 4, or "all"
    #[arg(short, long, default_value = "all")]
    scenario: String,

    /// Output directory for generated PNGs
    #[arg(short, long, default_value = "research/diagrams")]
    output_dir: PathBuf,

    /// Image width in pixels
    #[arg(long, default_value_t = 800)]
    width: u32,

    /// Image height in pixels
    #[arg(long, default_value_t = 400)]
    height: u32,
}

// ─── Paper-accurate simulation ─────────────────────────────────────────────

/// Scenario parameters matching the paper's §8 definitions.
struct Scenario {
    number: u8,
    title: &'static str,
    /// Discretized threshold T_d
    threshold: i32,
    /// Leak factor ℓ (0.0 to 1.0)
    leak_factor: f64,
    /// Number of threshold levels N
    threshold_levels: usize,
    /// Discretized input weight per step (δ_W(w))
    input_weight: i32,
    /// Simulation duration in ms
    duration_ms: f32,
    /// Time step dt in ms
    dt_ms: f32,
    /// RNG seed
    seed: u64,
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            number: 1,
            title: "Low Threshold (Td=3), Low Leak (ℓ=0.1)",
            threshold: 3,
            leak_factor: 0.1,
            threshold_levels: 4,
            input_weight: 3,
            duration_ms: 50.0,
            dt_ms: 0.1,
            seed: 42,
        },
        Scenario {
            number: 2,
            title: "Low Threshold (Td=3), High Leak (ℓ=0.5)",
            threshold: 3,
            leak_factor: 0.5,
            threshold_levels: 4,
            input_weight: 3,
            duration_ms: 50.0,
            dt_ms: 0.1,
            seed: 42,
        },
        Scenario {
            number: 3,
            title: "High Threshold (Td=6), Low Leak (ℓ=0.1)",
            threshold: 6,
            leak_factor: 0.1,
            threshold_levels: 4,
            input_weight: 3,
            duration_ms: 50.0,
            dt_ms: 0.1,
            seed: 42,
        },
        Scenario {
            number: 4,
            title: "High Threshold (Td=6), High Leak (ℓ=0.5)",
            threshold: 6,
            leak_factor: 0.5,
            threshold_levels: 4,
            input_weight: 3,
            duration_ms: 50.0,
            dt_ms: 0.1,
            seed: 42,
        },
    ]
}

/// Result of running the paper-accurate simulation.
struct PaperSimResult {
    /// Membrane potentials for the output neuron at each step
    potentials: Vec<i32>,
    /// Input neuron spike times (always fires)
    input_spikes: Vec<u32>,
    /// Output neuron spike times
    output_spikes: Vec<u32>,
    /// Total simulation steps
    total_steps: u32,
    /// Time step
    dt_ms: f32,
}

/// Run the paper's additive leak model:
///   p_{t+1} = max(0, p_t + λ_d + input)
///   λ_d = -max(1, floor(ℓ · T_d))
///
/// Firing uses N threshold levels with probabilities [1/N, 2/N, ..., 1.0].
fn run_paper_simulation(s: &Scenario) -> PaperSimResult {
    let total_steps = (s.duration_ms / s.dt_ms).round() as u32;
    let mut rng = StdRng::seed_from_u64(s.seed);

    // λ_d = -max(1, floor(ℓ · T_d))
    let lambda_d = -(1i32.max((s.leak_factor * s.threshold as f64).floor() as i32));

    // Generate threshold boundaries: [T_d*1/N, T_d*2/N, ..., T_d]
    let thresholds: Vec<i32> = (1..=s.threshold_levels)
        .map(|i| (i as i32 * s.threshold) / s.threshold_levels as i32)
        .collect();

    let mut potential: i32 = 0;
    let mut potentials = Vec::with_capacity(total_steps as usize);
    let mut input_spikes = Vec::new();
    let mut output_spikes = Vec::new();

    for step in 0..total_steps {
        // Record potential before update
        potentials.push(potential);

        // Input neuron always fires (AlwaysOn)
        input_spikes.push(step);

        // Paper update: p_{t+1} = max(0, p + λ_d + input)
        let new_potential = (potential + lambda_d + s.input_weight).max(0);

        // Determine firing probability from threshold levels
        let fire_prob = determine_fire_probability(new_potential, &thresholds);
        let fires = fire_prob > 0.0 && rng.r#gen::<f64>() < fire_prob;

        if fires {
            output_spikes.push(step);
            potential = 0; // Reset after spike
        } else {
            // Clamp to safety range
            potential = new_potential.clamp(0, s.threshold + 5);
        }
    }

    PaperSimResult {
        potentials,
        input_spikes,
        output_spikes,
        total_steps,
        dt_ms: s.dt_ms,
    }
}

/// Map potential to firing probability using threshold levels.
/// Returns 0.0 if below all thresholds, k/N if at level k.
fn determine_fire_probability(potential: i32, thresholds: &[i32]) -> f64 {
    let n = thresholds.len();
    // Count how many thresholds the potential exceeds or equals
    let level = thresholds.iter().filter(|&&t| potential >= t).count();
    level as f64 / n as f64
}

/// Run one scenario and render both PNGs.
fn run_scenario(
    scenario: &Scenario,
    output_dir: &std::path::Path,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let lambda_d = -(1i32.max((scenario.leak_factor * scenario.threshold as f64).floor() as i32));
    let net_gain = scenario.input_weight + lambda_d;

    println!("\n── Scenario {} : {} ──", scenario.number, scenario.title);
    println!("  λ_d = {}, net gain = {}", lambda_d, net_gain);

    let result = run_paper_simulation(scenario);

    let total_time_ms = result.total_steps as f32 * result.dt_ms;
    let input_rate = result.input_spikes.len() as f64 / (total_time_ms as f64 / 1000.0);
    let output_rate = result.output_spikes.len() as f64 / (total_time_ms as f64 / 1000.0);
    println!(
        "  Input:  {} spikes ({:.0} Hz)",
        result.input_spikes.len(),
        input_rate
    );
    println!(
        "  Output: {} spikes ({:.0} Hz)",
        result.output_spikes.len(),
        output_rate
    );
    println!("  Total steps: {}", result.total_steps);

    // Render raster plot
    let raster_path = output_dir.join(format!("diagram_{}_raster.png", scenario.number));
    draw_raster(&raster_path, &result, scenario, width, height)?;
    println!("  Raster    → {}", raster_path.display());

    // Render membrane potential plot
    let pot_path = output_dir.join(format!("diagram_{}_potential.png", scenario.number));
    draw_potential(&pot_path, &result, scenario, width, height)?;
    println!("  Potential → {}", pot_path.display());

    Ok(())
}

// ─── Raster plot ───────────────────────────────────────────────────────────

fn draw_raster(
    path: &std::path::Path,
    result: &PaperSimResult,
    scenario: &Scenario,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (width, height)).into_drawing_area();
    root.fill(&RGBColor(26, 26, 26))?;

    let total_ms = result.total_steps as f32 * result.dt_ms;
    let neuron_labels = ["Input", "Output"];
    let num_neurons = neuron_labels.len();

    let lambda_d = -(1i32.max((scenario.leak_factor * scenario.threshold as f64).floor() as i32));
    let subtitle = format!(
        "Td={}, ℓ={}, λd={}, w={}",
        scenario.threshold, scenario.leak_factor, lambda_d, scenario.input_weight
    );

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Diagram {} — Spike Raster", scenario.number),
            ("sans-serif", 18).into_font().color(&WHITE),
        )
        .margin(15)
        .margin_bottom(10)
        .x_label_area_size(35)
        .y_label_area_size(65)
        .build_cartesian_2d(0f32..total_ms, -0.5f32..(num_neurons as f32 - 0.5))?;

    chart
        .configure_mesh()
        .x_desc("Time (ms)")
        .y_desc("Neuron")
        .axis_desc_style(("sans-serif", 14).into_font().color(&WHITE))
        .label_style(("sans-serif", 12).into_font().color(&WHITE))
        .light_line_style(RGBColor(50, 50, 50))
        .bold_line_style(RGBColor(70, 70, 70))
        .y_labels(num_neurons)
        .y_label_formatter(&|y| {
            let idx = y.round() as usize;
            neuron_labels.get(idx).unwrap_or(&"").to_string()
        })
        .draw()?;

    // Draw spike markers for input (row 0) — orange
    for &step in &result.input_spikes {
        let t = step as f32 * result.dt_ms;
        chart.draw_series(std::iter::once(Circle::new(
            (t, 0.0f32),
            2,
            RGBColor(255, 150, 50).filled(),
        )))?;
    }

    // Draw spike markers for output (row 1) — cyan
    for &step in &result.output_spikes {
        let t = step as f32 * result.dt_ms;
        chart.draw_series(std::iter::once(Circle::new(
            (t, 1.0f32),
            3,
            RGBColor(100, 200, 255).filled(),
        )))?;
    }

    // Subtitle annotation
    chart.draw_series(std::iter::once(Text::new(
        subtitle,
        (total_ms * 0.02, num_neurons as f32 - 0.7),
        ("sans-serif", 11)
            .into_font()
            .color(&RGBColor(170, 170, 170)),
    )))?;

    root.present()?;
    Ok(())
}

// ─── Membrane potential plot ───────────────────────────────────────────────

fn draw_potential(
    path: &std::path::Path,
    result: &PaperSimResult,
    scenario: &Scenario,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (width, height)).into_drawing_area();
    root.fill(&RGBColor(26, 26, 26))?;

    let total_ms = result.total_steps as f32 * result.dt_ms;
    let dt_ms = result.dt_ms;

    // Figure out Y range from data
    let p_max = result
        .potentials
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(scenario.threshold);
    let y_min = -1.0f32;
    let y_max = (p_max as f32 + 2.0).max(scenario.threshold as f32 + 2.0);

    let lambda_d = -(1i32.max((scenario.leak_factor * scenario.threshold as f64).floor() as i32));
    let subtitle = format!(
        "Td={}, ℓ={}, λd={}, w={}",
        scenario.threshold, scenario.leak_factor, lambda_d, scenario.input_weight
    );

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Diagram {} — Membrane Potential (Output)", scenario.number),
            ("sans-serif", 18).into_font().color(&WHITE),
        )
        .margin(15)
        .margin_bottom(10)
        .x_label_area_size(35)
        .y_label_area_size(50)
        .build_cartesian_2d(0f32..total_ms, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Time (ms)")
        .y_desc("Potential (p)")
        .axis_desc_style(("sans-serif", 14).into_font().color(&WHITE))
        .label_style(("sans-serif", 12).into_font().color(&WHITE))
        .light_line_style(RGBColor(50, 50, 50))
        .bold_line_style(RGBColor(70, 70, 70))
        .draw()?;

    // Draw threshold line
    let threshold_f = scenario.threshold as f32;
    chart
        .draw_series(LineSeries::new(
            vec![(0.0, threshold_f), (total_ms, threshold_f)],
            RGBColor(255, 80, 80).stroke_width(2),
        ))?
        .label(format!("Td = {}", scenario.threshold))
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                RGBColor(255, 80, 80).stroke_width(2),
            )
        });

    // Draw sub-threshold level lines (dashed approximation using dotted segments)
    let thresholds: Vec<i32> = (1..scenario.threshold_levels)
        .map(|i| (i as i32 * scenario.threshold) / scenario.threshold_levels as i32)
        .collect();
    for &th in &thresholds {
        let th_f = th as f32;
        if th_f > y_min && th_f < y_max {
            // Draw as a dotted line
            let mut x = 0.0f32;
            let dash = total_ms * 0.01;
            let gap = total_ms * 0.01;
            let mut segments = Vec::new();
            while x < total_ms {
                let x_end = (x + dash).min(total_ms);
                segments.push((x, th_f));
                segments.push((x_end, th_f));
                x += dash + gap;
                // Break to separate segments
                segments.push((x, f32::NAN)); // NaN trick won't work with plotters
            }
            // Simple approach: draw as a thin solid line
            chart.draw_series(LineSeries::new(
                vec![(0.0, th_f), (total_ms, th_f)],
                RGBColor(100, 80, 80).stroke_width(1),
            ))?;
        }
    }

    // Draw potential trace
    let points: Vec<(f32, f32)> = result
        .potentials
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as f32 * dt_ms, p as f32))
        .collect();

    chart
        .draw_series(LineSeries::new(
            points,
            RGBColor(100, 200, 255).stroke_width(2),
        ))?
        .label("Output potential")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                RGBColor(100, 200, 255).stroke_width(2),
            )
        });

    // Mark spike events as red dots on the trace
    for &step in &result.output_spikes {
        let t = step as f32 * dt_ms;
        let p = result.potentials.get(step as usize).copied().unwrap_or(0) as f32;
        chart.draw_series(std::iter::once(Circle::new(
            (t, p),
            4,
            RGBColor(255, 80, 80).filled(),
        )))?;
    }

    chart
        .configure_series_labels()
        .background_style(RGBColor(40, 40, 40).mix(0.8))
        .border_style(RGBColor(80, 80, 80))
        .label_font(("sans-serif", 12).into_font().color(&WHITE))
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    // Subtitle
    chart.draw_series(std::iter::once(Text::new(
        subtitle,
        (total_ms * 0.02, y_max - 0.5),
        ("sans-serif", 11)
            .into_font()
            .color(&RGBColor(170, 170, 170)),
    )))?;

    root.present()?;
    Ok(())
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Ensure output directory exists
    std::fs::create_dir_all(&cli.output_dir)?;

    let all_scenarios = scenarios();

    let to_run: Vec<&Scenario> = if cli.scenario == "all" {
        all_scenarios.iter().collect()
    } else {
        let num: u8 = cli
            .scenario
            .parse()
            .map_err(|_| format!("Invalid scenario '{}': expected 1-4 or 'all'", cli.scenario))?;
        all_scenarios.iter().filter(|s| s.number == num).collect()
    };

    if to_run.is_empty() {
        eprintln!("No matching scenario found for '{}'", cli.scenario);
        std::process::exit(1);
    }

    println!("CogSpike Diagram Export (paper additive leak model)");
    println!("===================================================");
    println!("Output: {}", cli.output_dir.display());
    println!("Size:   {}x{}", cli.width, cli.height);

    for scenario in &to_run {
        run_scenario(scenario, &cli.output_dir, cli.width, cli.height)?;
    }

    println!("\nDone — {} diagram(s) exported.", to_run.len() * 2);
    Ok(())
}
