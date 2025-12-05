#![warn(clippy::all, rust_2018_idioms)]

mod app;
pub mod model_checker;
pub mod snn;
mod ui;

pub use app::TemplateApp;
