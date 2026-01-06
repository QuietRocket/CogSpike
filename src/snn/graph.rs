use rand::Rng as _;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub u32);

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct EdgeId(pub u32);

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeKind {
    Neuron,
}

impl NodeKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Neuron => "Neuron",
        }
    }

    /// Returns all node kinds available for manual creation.
    pub const fn palette() -> &'static [Self] {
        &[Self::Neuron]
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NeuronParams {
    /// Resting threshold potential (0-100, representing 0.0-1.0)
    pub p_rth: u8,
    /// Resting potential (0-100)
    pub p_rest: u8,
    /// Reset potential after spike (0-100)
    pub p_reset: u8,
    /// Leak rate (0-100, representing 0.0-1.0)
    pub leak_r: u8,
    /// Absolute refractory period (time steps)
    pub arp: u32,
    /// Relative refractory period (time steps)
    pub rrp: u32,
    /// Alpha scaling factor for RRP (0-100, representing 0.0-1.0)
    pub alpha: u8,
    /// Time step in tenths of ms (e.g., 10 = 1.0ms, 1 = 0.1ms)
    pub dt: u8,
    /// Firing probability thresholds (0-100 each, representing 0.0-1.0)
    pub thresholds: [u8; 10],
}

impl Default for NeuronParams {
    fn default() -> Self {
        Self {
            p_rth: 100,
            p_rest: 0,
            p_reset: 0,
            leak_r: 95,
            arp: 2,
            rrp: 4,
            alpha: 50,
            dt: 1, // 0.1ms in tenths
            thresholds: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    pub label: String,
    pub kind: NodeKind,
    pub params: NeuronParams,
    /// Optional learning target probability (0.0 to 1.0).
    /// When set, this node becomes a learning target.
    #[serde(default)]
    pub target_probability: Option<f64>,
    /// Optional PCTL formula for verification (e.g., "F \"spike\"").
    #[serde(default)]
    pub target_formula: Option<String>,
    pub position: [f32; 2],
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Edge {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    /// Weight magnitude (0-100, representing 0.0-1.0).
    pub weight: u8,
    /// If true, this is an inhibitory synapse; otherwise it's excitatory.
    #[serde(default)]
    pub is_inhibitory: bool,
}

impl Edge {
    /// Get the effective signed weight (negative if inhibitory).
    /// Returns i16 to safely handle the full range of -100 to +100.
    pub fn signed_weight(&self) -> i16 {
        let w = self.weight as i16;
        if self.is_inhibitory { -w } else { w }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SnnGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    #[serde(default = "SnnGraph::default_node_id")]
    next_node_id: u32,
    #[serde(default = "SnnGraph::default_edge_id")]
    next_edge_id: u32,
}

impl Default for SnnGraph {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            next_node_id: Self::default_node_id(),
            next_edge_id: Self::default_edge_id(),
        }
    }
}

impl SnnGraph {
    const DEFAULT_WEIGHT: u8 = 100;

    const fn default_node_id() -> u32 {
        1
    }

    const fn default_edge_id() -> u32 {
        1
    }

    pub fn add_node(
        &mut self,
        label: impl Into<String>,
        kind: NodeKind,
        position: [f32; 2],
    ) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        self.nodes.push(Node {
            id,
            label: label.into(),
            kind,
            params: NeuronParams::default(),
            target_probability: None,
            target_formula: None,
            position,
        });
        id
    }

    /// Add an edge between two nodes with the given weight magnitude (0-100).
    /// The `is_inhibitory` flag defaults to false (excitatory).
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, weight: u8) -> Option<EdgeId> {
        self.add_edge_with_type(from, to, weight, false)
    }

    /// Add an inhibitory edge between two nodes.
    pub fn add_edge_inhibitory(&mut self, from: NodeId, to: NodeId, weight: u8) -> Option<EdgeId> {
        self.add_edge_with_type(from, to, weight, true)
    }

    /// Add an edge with explicit excitatory/inhibitory type.
    fn add_edge_with_type(
        &mut self,
        from: NodeId,
        to: NodeId,
        weight: u8,
        is_inhibitory: bool,
    ) -> Option<EdgeId> {
        if from == to {
            return None;
        }
        let exists = self
            .edges
            .iter()
            .any(|edge| edge.from == from && edge.to == to);
        if exists {
            return None;
        }
        let id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;
        self.edges.push(Edge {
            id,
            from,
            to,
            weight: weight.min(100),
            is_inhibitory,
        });
        Some(id)
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn remove_node(&mut self, id: NodeId) {
        self.nodes.retain(|n| n.id != id);
        self.edges.retain(|e| e.from != id && e.to != id);
    }

    pub fn remove_edge(&mut self, id: EdgeId) {
        self.edges.retain(|e| e.id != id);
    }

    pub fn position_of(&self, id: NodeId) -> Option<[f32; 2]> {
        self.node(id).map(|n| n.position)
    }

    /// Returns nodes with no incoming edges (root nodes / input generators).
    pub fn input_neurons(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|n| !self.edges.iter().any(|e| e.to == n.id))
            .map(|n| n.id)
            .collect()
    }

    /// Returns nodes with no outgoing edges (leaf nodes / outputs).
    pub fn output_neurons(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|n| !self.edges.iter().any(|e| e.from == n.id))
            .map(|n| n.id)
            .collect()
    }

    /// Returns all edges pointing to the given node.
    pub fn incoming_edges(&self, id: NodeId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.to == id).collect()
    }

    /// Returns all edges originating from the given node.
    pub fn outgoing_edges(&self, id: NodeId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.from == id).collect()
    }

    /// Check if node is topologically an input (no incoming edges).
    pub fn is_input(&self, id: NodeId) -> bool {
        self.node(id).is_some() && !self.edges.iter().any(|e| e.to == id)
    }

    /// Check if node is topologically an output (no outgoing edges).
    pub fn is_output(&self, id: NodeId) -> bool {
        self.node(id).is_some() && !self.edges.iter().any(|e| e.from == id)
    }

    /// Check if a node is an input generator (topologically an input).
    pub fn is_input_generator(&self, id: NodeId) -> bool {
        self.is_input(id)
    }

    /// Returns all nodes that are learning targets (have `target_probability` set).
    pub fn learning_targets(&self) -> Vec<&Node> {
        self.nodes
            .iter()
            .filter(|n| n.target_probability.is_some())
            .collect()
    }

    /// Find the edge by ID and return an immutable reference.
    pub fn edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges.iter().find(|e| e.id == id)
    }

    /// Find the edge by ID and return a mutable reference.
    pub fn edge_mut(&mut self, id: EdgeId) -> Option<&mut Edge> {
        self.edges.iter_mut().find(|e| e.id == id)
    }

    /// Update the weight magnitude of an edge, clamping to [0, 100].
    pub fn update_weight(&mut self, id: EdgeId, new_weight: u8) {
        if let Some(edge) = self.edge_mut(id) {
            edge.weight = new_weight.min(100);
        }
    }

    pub fn demo_layout() -> Self {
        let mut graph = Self::default();
        let input = graph.add_node("Input", NodeKind::Neuron, [80.0, 120.0]);
        let neuron_a = graph.add_node("Neuron A", NodeKind::Neuron, [220.0, 140.0]);
        let neuron_b = graph.add_node("Neuron B", NodeKind::Neuron, [360.0, 110.0]);
        let output = graph.add_node("Output", NodeKind::Neuron, [520.0, 200.0]);

        graph.add_edge(input, neuron_a, Self::DEFAULT_WEIGHT);
        graph.add_edge(neuron_a, neuron_b, Self::DEFAULT_WEIGHT);
        // Use inhibitory edge for the last connection
        graph.add_edge_inhibitory(neuron_b, output, Self::DEFAULT_WEIGHT);

        graph
    }

    /// Randomize all edge weight magnitudes within the given range [min, max].
    /// Weights are clamped to [0, 100] after randomization.
    pub fn randomize_weights(&mut self, min: u8, max: u8) {
        let mut rng = rand::thread_rng();
        let min_val = min.min(100);
        let max_val = max.min(100);
        for edge in &mut self.edges {
            edge.weight = rng.gen_range(min_val..=max_val);
        }
    }

    /// Randomize all edge weight magnitudes within [0, range].
    pub fn randomize_weights_symmetric(&mut self, range: u8) {
        self.randomize_weights(0, range);
    }
}
