use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub u32);

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct EdgeId(pub u32);

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeKind {
    Neuron,
    Population,
    Input,
    Output,
}

impl NodeKind {
    pub fn label(self) -> &'static str {
        match self {
            NodeKind::Neuron => "Neuron",
            NodeKind::Population => "Population",
            NodeKind::Input => "Input",
            NodeKind::Output => "Output",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NeuronParams {
    pub p_rth: f32,
    pub p_rest: f32,
    pub p_reset: f32,
    pub leak_r: f32,
    pub arp: u32,
    pub rrp: u32,
    pub alpha: f32,
    pub dt: f32,
    pub thresholds: [f32; 10],
}

impl Default for NeuronParams {
    fn default() -> Self {
        Self {
            p_rth: 1.0,
            p_rest: 0.0,
            p_reset: 0.0,
            leak_r: 0.95,
            arp: 2,
            rrp: 4,
            alpha: 0.5,
            dt: 0.1,
            thresholds: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    pub label: String,
    pub kind: NodeKind,
    pub params: NeuronParams,
    pub position: [f32; 2],
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Edge {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub weight: f32,
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
    const DEFAULT_WEIGHT: f32 = 1.0;

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
            position,
        });
        id
    }

    pub fn add_edge(&mut self, from: NodeId, to: NodeId, weight: f32) -> Option<EdgeId> {
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
            weight,
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

    /// Check if a node is an input generator (ignores learning advice).
    pub fn is_input_generator(&self, id: NodeId) -> bool {
        self.node(id).map_or(false, |n| n.kind == NodeKind::Input)
    }

    pub fn demo_layout() -> Self {
        let mut graph = Self::default();
        let input = graph.add_node("Input", NodeKind::Input, [80.0, 120.0]);
        let neuron_a = graph.add_node("Neuron A", NodeKind::Neuron, [220.0, 140.0]);
        let neuron_b = graph.add_node("Neuron B", NodeKind::Neuron, [360.0, 110.0]);
        let output = graph.add_node("Output", NodeKind::Output, [520.0, 200.0]);

        graph.add_edge(input, neuron_a, Self::DEFAULT_WEIGHT);
        graph.add_edge(neuron_a, neuron_b, Self::DEFAULT_WEIGHT);
        graph.add_edge(neuron_b, output, -Self::DEFAULT_WEIGHT);

        graph
    }
}
