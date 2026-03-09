//! OOM benchmark: generates PRISM models for chains and forks at varying sizes.
//!
//! For each (topology, config_preset, model_type, W) combination, this test
//! generates `.pm` files for sizes 1..MAX_SIZE.  The companion shell script
//! `research/limits/run_benchmark.sh` then probes these with PRISM using
//! binary search to find the maximal size before OOM.
//!
//! Run:  `cargo test --test benchmark_limits -- --nocapture`
//! Output: `research/limits/benchmark/*.pm`

use cog_spike::simulation::{InputNeuronConfig, ModelConfig};
use cog_spike::snn::graph::{NodeKind, SnnGraph};
use cog_spike::snn::prism_discretized_gen::generate_discretized_model;
use cog_spike::snn::prism_gen::{PrismGenConfig, generate_prism_model};
use std::fs;

/// Maximum topology size to pre-generate.
/// The binary-search script only invokes a subset of these.
const MAX_CHAIN: usize = 30;
const MAX_FORK: usize = 30;

// ---------------------------------------------------------------------------
// Topology builders
// ---------------------------------------------------------------------------

/// Build a chain of `n` processing neurons: In → N1 → N2 → … → Nn.
fn build_chain(n: usize) -> SnnGraph {
    let mut g = SnnGraph::default();
    let inp = g.add_node("In", NodeKind::Neuron, [0.0, 0.0]);

    let mut prev = inp;
    for i in 1..=n {
        let ni = g.add_node(&format!("N{i}"), NodeKind::Neuron, [i as f32, 0.0]);
        g.add_edge(prev, ni, 80);
        prev = ni;
    }

    configure_inputs(&mut g);
    g
}

/// Build a fork with `b` branches: In → N1 → {N2, N3, …, N_{b+1}}.
/// There is always one "hub" neuron N1 that fans out to `b` leaf neurons.
fn build_fork(b: usize) -> SnnGraph {
    let mut g = SnnGraph::default();
    let inp = g.add_node("In", NodeKind::Neuron, [0.0, 0.0]);
    let hub = g.add_node("N1", NodeKind::Neuron, [1.0, 0.0]);
    g.add_edge(inp, hub, 80);

    for i in 0..b {
        let y = (i as f32) - (b as f32 - 1.0) / 2.0;
        let leaf = g.add_node(&format!("N{}", i + 2), NodeKind::Neuron, [2.0, y]);
        g.add_edge(hub, leaf, 80);
    }

    configure_inputs(&mut g);
    g
}

/// Configure all input neurons with AlwaysOn.
fn configure_inputs(graph: &mut SnnGraph) {
    let inputs = graph.input_neurons();
    for id in inputs {
        if let Some(node) = graph.node_mut(id) {
            node.input_config = Some(InputNeuronConfig::with_default_generator());
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration helpers
// ---------------------------------------------------------------------------

struct ConfigPreset {
    name: &'static str,
    model: ModelConfig,
}

fn presets() -> Vec<ConfigPreset> {
    vec![
        ConfigPreset {
            name: "deterministic",
            model: ModelConfig::deterministic(),
        },
        ConfigPreset {
            name: "fast",
            model: ModelConfig::default(), // 4 thresholds, no ARP/RRP
        },
        ConfigPreset {
            name: "full",
            model: ModelConfig::full(), // 10 thresholds, ARP+RRP
        },
    ]
}

fn make_gen_config(model: ModelConfig, weight_levels: u8) -> PrismGenConfig {
    PrismGenConfig {
        model,
        weight_levels,
        time_bound: Some(20),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Test entry point
// ---------------------------------------------------------------------------

#[test]
fn generate_benchmark_models() {
    let out_dir = std::path::Path::new("research/limits/benchmark");
    fs::create_dir_all(out_dir).expect("create benchmark directory");

    let weight_levels: &[u8] = &[2, 3, 5];
    let mut file_count: usize = 0;

    for preset in presets() {
        // ── Chains ──────────────────────────────────────────────────
        for n in 1..=MAX_CHAIN {
            let graph = build_chain(n);

            // Precise
            let cfg = make_gen_config(preset.model.clone(), 3);
            let pm = generate_prism_model(&graph, &cfg);
            let fname = format!("chain{}_{}_precise.pm", n, preset.name);
            fs::write(out_dir.join(&fname), &pm).unwrap();
            file_count += 1;

            // Discretized at each W
            for &w in weight_levels {
                let cfg = make_gen_config(preset.model.clone(), w);
                let pm = generate_discretized_model(&graph, &cfg);
                let fname = format!("chain{}_{}_disc_w{}.pm", n, preset.name, w);
                fs::write(out_dir.join(&fname), &pm).unwrap();
                file_count += 1;
            }
        }

        // ── Forks ───────────────────────────────────────────────────
        for b in 1..=MAX_FORK {
            let graph = build_fork(b);

            // Precise
            let cfg = make_gen_config(preset.model.clone(), 3);
            let pm = generate_prism_model(&graph, &cfg);
            let fname = format!("fork{}_{}_precise.pm", b, preset.name);
            fs::write(out_dir.join(&fname), &pm).unwrap();
            file_count += 1;

            // Discretized at each W
            for &w in weight_levels {
                let cfg = make_gen_config(preset.model.clone(), w);
                let pm = generate_discretized_model(&graph, &cfg);
                let fname = format!("fork{}_{}_disc_w{}.pm", b, preset.name, w);
                fs::write(out_dir.join(&fname), &pm).unwrap();
                file_count += 1;
            }
        }
    }

    println!(
        "\n=== Generated {file_count} .pm files in {} ===",
        out_dir.display()
    );
    println!("Next: run  bash research/limits/run_benchmark.sh");
}
