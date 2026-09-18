//! Versioned, bounded frames. No inference dependencies belong in this crate.
#[cfg(windows)]
pub mod client;
#[cfg(windows)]
pub mod pipe;
pub mod protocol;
