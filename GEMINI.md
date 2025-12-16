## EGui Considerations

Since EGui is immediate mode, the way interactions and display layout must be implemented should be immediate mode friendly.
If you ever have doubts on using EGui or EFrame, refer to the documentation using Context7.

## General Rust Guide

Always perform a `cargo check` before terminating a task to make sure there are no issues with the Rust compiler.

## Warning-Free Code

At the end of each large series of modifications, ensure there are **no warnings** from the Rust compiler. Run `cargo build --release` or `cargo check` and address all warnings before considering the work complete. This includes:
- Unused imports
- Unused variables
- Dead code
- Deprecated API usage
- Any other compiler warnings
