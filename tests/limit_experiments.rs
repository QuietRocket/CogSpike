//! Integration tests for SNN topology limit analysis.
//!
//! Generates PRISM `.pm` files for canonical topologies (single-neuron, chain,
//! fork, diamond) with both precise and discretized models, in fast and full
//! ModelConfig presets.
//!
//! Run: `cargo test --test limit_experiments -- --nocapture`
//! Generated files land in: `research/limits/experiments/`

use cog_spike::simulation::{InputNeuronConfig, ModelConfig};
use cog_spike::snn::graph::{NodeKind, SnnGraph};
use cog_spike::snn::prism_discretized_gen::generate_discretized_model;
use cog_spike::snn::prism_gen::{PrismGenConfig, generate_prism_model};
use std::fs;

/// Helper: build a graph and configure input neurons with AlwaysOn pattern.
fn configure_inputs(graph: &mut SnnGraph) {
    let inputs = graph.input_neurons();
    for id in inputs {
        if let Some(node) = graph.node_mut(id) {
            let cfg = InputNeuronConfig::with_default_generator();
            node.input_config = Some(cfg);
        }
    }
}

/// Helper: build PrismGenConfig from a ModelConfig preset.
fn make_config(model: ModelConfig, weight_levels: u8) -> PrismGenConfig {
    PrismGenConfig {
        model,
        weight_levels,
        time_bound: Some(20), // Keep small for fast export
        ..Default::default()
    }
}

/// Build a single-neuron topology: 1 input → 1 neuron.
fn topo_single() -> SnnGraph {
    let mut g = SnnGraph::default();
    let inp = g.add_node("In", NodeKind::Neuron, [0.0, 0.0]);
    let n1 = g.add_node("N1", NodeKind::Neuron, [1.0, 0.0]);
    g.add_edge(inp, n1, 80);
    configure_inputs(&mut g);
    g
}

/// Build a 2-neuron chain: 1 input → N1 → N2.
fn topo_chain2() -> SnnGraph {
    let mut g = SnnGraph::default();
    let inp = g.add_node("In", NodeKind::Neuron, [0.0, 0.0]);
    let n1 = g.add_node("N1", NodeKind::Neuron, [1.0, 0.0]);
    let n2 = g.add_node("N2", NodeKind::Neuron, [2.0, 0.0]);
    g.add_edge(inp, n1, 80);
    g.add_edge(n1, n2, 80);
    configure_inputs(&mut g);
    g
}

/// Build a 3-neuron chain: 1 input → N1 → N2 → N3.
fn topo_chain3() -> SnnGraph {
    let mut g = SnnGraph::default();
    let inp = g.add_node("In", NodeKind::Neuron, [0.0, 0.0]);
    let n1 = g.add_node("N1", NodeKind::Neuron, [1.0, 0.0]);
    let n2 = g.add_node("N2", NodeKind::Neuron, [2.0, 0.0]);
    let n3 = g.add_node("N3", NodeKind::Neuron, [3.0, 0.0]);
    g.add_edge(inp, n1, 80);
    g.add_edge(n1, n2, 80);
    g.add_edge(n2, n3, 80);
    configure_inputs(&mut g);
    g
}

/// Build a fork: 1 input → N1 → {N2, N3}.
fn topo_fork() -> SnnGraph {
    let mut g = SnnGraph::default();
    let inp = g.add_node("In", NodeKind::Neuron, [0.0, 0.0]);
    let n1 = g.add_node("N1", NodeKind::Neuron, [1.0, 0.0]);
    let n2 = g.add_node("N2", NodeKind::Neuron, [2.0, -1.0]);
    let n3 = g.add_node("N3", NodeKind::Neuron, [2.0, 1.0]);
    g.add_edge(inp, n1, 80);
    g.add_edge(n1, n2, 80);
    g.add_edge(n1, n3, 80);
    configure_inputs(&mut g);
    g
}

/// Build a diamond: 1 input → N1 → {N2, N3} → N4.
fn topo_diamond() -> SnnGraph {
    let mut g = SnnGraph::default();
    let inp = g.add_node("In", NodeKind::Neuron, [0.0, 0.0]);
    let n1 = g.add_node("N1", NodeKind::Neuron, [1.0, 0.0]);
    let n2 = g.add_node("N2", NodeKind::Neuron, [2.0, -1.0]);
    let n3 = g.add_node("N3", NodeKind::Neuron, [2.0, 1.0]);
    let n4 = g.add_node("N4", NodeKind::Neuron, [3.0, 0.0]);
    g.add_edge(inp, n1, 80);
    g.add_edge(n1, n2, 80);
    g.add_edge(n1, n3, 80);
    g.add_edge(n2, n4, 80);
    g.add_edge(n3, n4, 80);
    configure_inputs(&mut g);
    g
}

/// Build a 4-neuron chain for scaling observation.
fn topo_chain4() -> SnnGraph {
    let mut g = SnnGraph::default();
    let inp = g.add_node("In", NodeKind::Neuron, [0.0, 0.0]);
    let n1 = g.add_node("N1", NodeKind::Neuron, [1.0, 0.0]);
    let n2 = g.add_node("N2", NodeKind::Neuron, [2.0, 0.0]);
    let n3 = g.add_node("N3", NodeKind::Neuron, [3.0, 0.0]);
    let n4 = g.add_node("N4", NodeKind::Neuron, [4.0, 0.0]);
    g.add_edge(inp, n1, 80);
    g.add_edge(n1, n2, 80);
    g.add_edge(n2, n3, 80);
    g.add_edge(n3, n4, 80);
    configure_inputs(&mut g);
    g
}

/// Build a 2-input convergent topology: 2 inputs → N1.
fn topo_convergent2() -> SnnGraph {
    let mut g = SnnGraph::default();
    let in1 = g.add_node("In1", NodeKind::Neuron, [0.0, -1.0]);
    let in2 = g.add_node("In2", NodeKind::Neuron, [0.0, 1.0]);
    let n1 = g.add_node("N1", NodeKind::Neuron, [1.0, 0.0]);
    g.add_edge(in1, n1, 60);
    g.add_edge(in2, n1, 60);
    configure_inputs(&mut g);
    g
}

struct TopologyDef {
    name: &'static str,
    builder: fn() -> SnnGraph,
}

const TOPOLOGIES: &[TopologyDef] = &[
    TopologyDef {
        name: "single",
        builder: topo_single,
    },
    TopologyDef {
        name: "chain2",
        builder: topo_chain2,
    },
    TopologyDef {
        name: "chain3",
        builder: topo_chain3,
    },
    TopologyDef {
        name: "chain4",
        builder: topo_chain4,
    },
    TopologyDef {
        name: "fork",
        builder: topo_fork,
    },
    TopologyDef {
        name: "diamond",
        builder: topo_diamond,
    },
    TopologyDef {
        name: "convergent2",
        builder: topo_convergent2,
    },
];

struct ConfigPreset {
    name: &'static str,
    model: ModelConfig,
}

fn presets() -> Vec<ConfigPreset> {
    vec![
        ConfigPreset {
            name: "fast",
            model: ModelConfig::default(), // 4 thresholds, no ARP/RRP
        },
        ConfigPreset {
            name: "full",
            model: ModelConfig::full(), // 10 thresholds, ARP+RRP
        },
        ConfigPreset {
            name: "deterministic",
            model: ModelConfig::deterministic(), // 1 threshold, no ARP/RRP
        },
    ]
}

#[test]
fn generate_all_experiment_models() {
    let out_dir = std::path::Path::new("research/limits/experiments");
    fs::create_dir_all(out_dir).expect("Failed to create experiments directory");

    let mut summary = String::from(
        "topology,config,model_type,weight_levels,neurons,internal_neurons,edges,internal_edges,filename\n",
    );

    for topo in TOPOLOGIES {
        let graph = (topo.builder)();
        let n_neurons = graph.nodes.len();
        let n_inputs = graph.input_neurons().len();
        let n_internal = n_neurons - n_inputs;
        let n_edges = graph.edges.len();
        let n_internal_edges = graph
            .edges
            .iter()
            .filter(|e| !graph.is_input(e.from))
            .count();

        for preset in presets() {
            // --- Precise model ---
            let precise_config = make_config(preset.model.clone(), 3);
            let precise_pm = generate_prism_model(&graph, &precise_config);
            let precise_fname = format!("{}_{}_precise.pm", topo.name, preset.name);
            let precise_path = out_dir.join(&precise_fname);
            fs::write(&precise_path, &precise_pm)
                .unwrap_or_else(|e| panic!("Failed to write {precise_fname}: {e}"));
            summary.push_str(&format!(
                "{},{},precise,N/A,{},{},{},{},{}\n",
                topo.name,
                preset.name,
                n_neurons,
                n_internal,
                n_edges,
                n_internal_edges,
                precise_fname
            ));

            // --- Discretized models with W=2 and W=3 ---
            for wl in [2u8, 3] {
                let disc_config = make_config(preset.model.clone(), wl);
                let disc_pm = generate_discretized_model(&graph, &disc_config);
                let disc_fname = format!("{}_{}_disc_w{}.pm", topo.name, preset.name, wl);
                let disc_path = out_dir.join(&disc_fname);
                fs::write(&disc_path, &disc_pm)
                    .unwrap_or_else(|e| panic!("Failed to write {disc_fname}: {e}"));
                summary.push_str(&format!(
                    "{},{},discretized,{},{},{},{},{},{}\n",
                    topo.name,
                    preset.name,
                    wl,
                    n_neurons,
                    n_internal,
                    n_edges,
                    n_internal_edges,
                    disc_fname
                ));
            }
        }
    }

    // Write summary CSV
    let summary_path = out_dir.join("model_summary.csv");
    fs::write(&summary_path, &summary).expect("Failed to write summary CSV");

    println!(
        "\n=== Generated {} .pm files ===",
        TOPOLOGIES.len() * presets().len() * 3
    );
    println!("Summary: {}", summary_path.display());
    println!("\nNext: run PRISM exports with:");
    println!("  cd research/limits/experiments && bash run_prism_exports.sh");
}
