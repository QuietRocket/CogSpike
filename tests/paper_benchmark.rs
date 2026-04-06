//! Paper benchmark: generates PRISM models for the paper's Table 1 and case study.
//!
//! Generates `.pm` files for 7 canonical topologies + contralateral inhibition,
//! comparing precise models vs discretized W=6 models with multiplicative leak.
//!
//! Focus: fast preset (k=4 threshold levels, no refractory period) as requested.
//!
//! Run:  `cargo test --test paper_benchmark -- --nocapture`
//! Output: `benchmark/*.pm` + `benchmark/model_summary.csv`

use cog_spike::simulation::{InputNeuronConfig, ModelConfig};
use cog_spike::snn::graph::{NodeKind, NodeRole, SnnGraph};
use cog_spike::snn::prism_discretized_gen::generate_discretized_model;
use cog_spike::snn::prism_gen::{generate_prism_model, PrismGenConfig};
use std::fs;

// ============================================================================
// Topology Builders
// ============================================================================

/// Configure all input neurons with AlwaysOn.
fn configure_inputs(graph: &mut SnnGraph) {
    let inputs = graph.input_neurons();
    for id in inputs {
        if let Some(node) = graph.node_mut(id) {
            node.input_config = Some(InputNeuronConfig::with_default_generator());
        }
    }
}

/// Single neuron: In → N1.
fn topo_single() -> SnnGraph {
    let mut g = SnnGraph::default();
    let inp = g.add_node("In", NodeKind::Neuron, [0.0, 0.0]);
    let n1 = g.add_node("N1", NodeKind::Neuron, [1.0, 0.0]);
    g.add_edge(inp, n1, 80);
    configure_inputs(&mut g);
    g
}

/// 2-neuron chain: In → N1 → N2.
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

/// 3-neuron chain: In → N1 → N2 → N3.
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

/// 4-neuron chain: In → N1 → N2 → N3 → N4.
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

/// Fork: In → N1 → {N2, N3}.
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

/// Diamond: In → N1 → {N2, N3} → N4.
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

/// Convergent: {In1, In2} → N1 (fan-in 2, subthreshold weights 60 each).
fn topo_convergent() -> SnnGraph {
    let mut g = SnnGraph::default();
    let in1 = g.add_node("In1", NodeKind::Neuron, [0.0, -1.0]);
    let in2 = g.add_node("In2", NodeKind::Neuron, [0.0, 1.0]);
    let n1 = g.add_node("N1", NodeKind::Neuron, [1.0, 0.0]);
    g.add_edge(in1, n1, 60);
    g.add_edge(in2, n1, 60);
    configure_inputs(&mut g);
    g
}

/// 3-channel contralateral inhibition matching paper's case study (§7).
///
/// 9 neurons total: S1–S3 (input), N1–N3 (processing), O1–O3 (output).
/// Inhibitory weights: N1→{N2,N3} = -100 (winner), N2→{N1,N3} = -70,
/// N3→{N1,N2} = -70. Excitatory: S→N = +100, N→O = +100.
///
/// Uses NodeRole::Input/Output to handle recurrent topology correctly.
fn topo_contra3() -> SnnGraph {
    let mut g = SnnGraph::default();

    // Input neurons
    let s1 = g.add_node("S1", NodeKind::Neuron, [0.0, -1.0]);
    let s2 = g.add_node("S2", NodeKind::Neuron, [0.0, 0.0]);
    let s3 = g.add_node("S3", NodeKind::Neuron, [0.0, 1.0]);

    // Processing neurons (competitors)
    let n1 = g.add_node("N1", NodeKind::Neuron, [1.0, -1.0]);
    let n2 = g.add_node("N2", NodeKind::Neuron, [1.0, 0.0]);
    let n3 = g.add_node("N3", NodeKind::Neuron, [1.0, 1.0]);

    // Output neurons
    let o1 = g.add_node("O1", NodeKind::Neuron, [2.0, -1.0]);
    let o2 = g.add_node("O2", NodeKind::Neuron, [2.0, 0.0]);
    let o3 = g.add_node("O3", NodeKind::Neuron, [2.0, 1.0]);

    // Set explicit roles (required for recurrent topology)
    for id in [s1, s2, s3] {
        g.node_mut(id).unwrap().role = NodeRole::Input;
    }
    for id in [o1, o2, o3] {
        g.node_mut(id).unwrap().role = NodeRole::Output;
    }

    // Excitatory: input → processing (w = +100)
    g.add_edge(s1, n1, 100);
    g.add_edge(s2, n2, 100);
    g.add_edge(s3, n3, 100);

    // Excitatory: processing → output (w = +100)
    g.add_edge(n1, o1, 100);
    g.add_edge(n2, o2, 100);
    g.add_edge(n3, o3, 100);

    // Inhibitory: N1 → N2, N3 (strong, w = -100, winner)
    g.add_edge_inhibitory(n1, n2, 100);
    g.add_edge_inhibitory(n1, n3, 100);

    // Inhibitory: N2 → N1, N3 (weak, w = -70)
    g.add_edge_inhibitory(n2, n1, 70);
    g.add_edge_inhibitory(n2, n3, 70);

    // Inhibitory: N3 → N1, N2 (weak, w = -70)
    g.add_edge_inhibitory(n3, n1, 70);
    g.add_edge_inhibitory(n3, n2, 70);

    configure_inputs(&mut g);
    g
}

// ============================================================================
// Config Helpers
// ============================================================================

/// Paper case study configuration: k=4, p_rth=80, leak_r=50 (r=0.5), no ARP/RRP.
fn case_study_config() -> ModelConfig {
    ModelConfig {
        threshold_levels: 4,
        enable_arp: false,
        enable_rrp: false,
        p_rth: 80,
        leak_r: 50, // r = 0.5
        prism_potential_threshold: None,
        ..Default::default()
    }
}

/// Fast preset: k=4, no ARP/RRP (default ModelConfig).
fn fast_config() -> ModelConfig {
    ModelConfig::default()
}

/// Deterministic preset: k=1, no ARP/RRP.
fn det_config() -> ModelConfig {
    ModelConfig::deterministic()
}

fn make_prism_config(model: ModelConfig, weight_levels: u8) -> PrismGenConfig {
    PrismGenConfig {
        model,
        weight_levels,
        time_bound: Some(100),
        ..Default::default()
    }
}

// ============================================================================
// Test Entry Point
// ============================================================================

struct TopologyDef {
    name: &'static str,
    builder: fn() -> SnnGraph,
    /// Use case study config (p_rth=80, r=0.5) instead of default fast config.
    use_case_study_config: bool,
}

#[test]
fn generate_paper_benchmark_models() {
    let out_dir = std::path::Path::new("benchmark");
    fs::create_dir_all(out_dir).expect("create benchmark directory");

    let topologies = vec![
        TopologyDef { name: "single",      builder: topo_single,     use_case_study_config: false },
        TopologyDef { name: "chain2",      builder: topo_chain2,     use_case_study_config: false },
        TopologyDef { name: "chain3",      builder: topo_chain3,     use_case_study_config: false },
        TopologyDef { name: "chain4",      builder: topo_chain4,     use_case_study_config: false },
        TopologyDef { name: "fork",        builder: topo_fork,       use_case_study_config: false },
        TopologyDef { name: "diamond",     builder: topo_diamond,    use_case_study_config: false },
        TopologyDef { name: "convergent",  builder: topo_convergent, use_case_study_config: false },
        TopologyDef { name: "contra3",     builder: topo_contra3,    use_case_study_config: true  },
    ];

    let mut summary = String::from(
        "topology,config,model_type,weight_levels,total_neurons,processing_neurons,total_edges,internal_edges,filename\n",
    );

    let mut file_count: usize = 0;

    for topo in &topologies {
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

        // Determine configs to use
        let configs: Vec<(&str, ModelConfig)> = if topo.use_case_study_config {
            vec![("casestudy", case_study_config())]
        } else {
            vec![
                ("det", det_config()),
                ("fast", fast_config()),
            ]
        };

        for (config_name, model_config) in &configs {
            // ── Precise model ──────────────────────────────────────────
            let cfg = make_prism_config(model_config.clone(), 6);
            let pm = generate_prism_model(&graph, &cfg);
            let fname = format!("{}_{}_precise.pm", topo.name, config_name);
            fs::write(out_dir.join(&fname), &pm).unwrap();
            summary.push_str(&format!(
                "{},{},precise,N/A,{},{},{},{},{}\n",
                topo.name, config_name, n_neurons, n_internal, n_edges, n_internal_edges, fname
            ));
            file_count += 1;

            // ── Discretized W=6 model ──────────────────────────────────
            let cfg = make_prism_config(model_config.clone(), 6);
            let pm = generate_discretized_model(&graph, &cfg);
            let fname = format!("{}_{}_disc_w6.pm", topo.name, config_name);
            fs::write(out_dir.join(&fname), &pm).unwrap();
            summary.push_str(&format!(
                "{},{},discretized,6,{},{},{},{},{}\n",
                topo.name, config_name, n_neurons, n_internal, n_edges, n_internal_edges, fname
            ));
            file_count += 1;

            // ── Discretized W=3 model (for comparison) ─────────────────
            let cfg = make_prism_config(model_config.clone(), 3);
            let pm = generate_discretized_model(&graph, &cfg);
            let fname = format!("{}_{}_disc_w3.pm", topo.name, config_name);
            fs::write(out_dir.join(&fname), &pm).unwrap();
            summary.push_str(&format!(
                "{},{},discretized,3,{},{},{},{},{}\n",
                topo.name, config_name, n_neurons, n_internal, n_edges, n_internal_edges, fname
            ));
            file_count += 1;
        }
    }

    // Write summary CSV
    let summary_path = out_dir.join("model_summary.csv");
    fs::write(&summary_path, &summary).expect("write summary CSV");

    println!("\n=== Generated {file_count} .pm files in {} ===", out_dir.display());
    println!("Summary: {}", summary_path.display());
    println!("\nNext: run  bash benchmark/run_prism.sh");
}
