//! Terminal capability profile and bounded live queries.
//!
//! [`profile::TermProfile`] is the pure environment snapshot used for terminal
//! capability decisions. Live terminal I/O stays isolated in [`background`].

pub mod background;
pub mod frame_writer;
pub mod profile;

pub use background::detect_background;
pub use frame_writer::FrameWriter;
pub use profile::{TermProfile, reduce_motion_enabled};
