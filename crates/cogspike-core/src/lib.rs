#![warn(clippy::all, rust_2018_idioms)]

//! cogspike-core: the egui-free simulation, learning, formal-verification, and
//! (forthcoming) latency-coder / gym library behind the CogSpike workbench.

pub mod coder;
pub mod gym;
pub mod learning;
pub mod model_checker;
pub mod net;
pub mod simulation;
pub mod snn;
pub mod substrate;
