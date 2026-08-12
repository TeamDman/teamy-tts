//! The single inference backend used by teamy-tts main.

use eyre::Result;
use std::fmt;

/// The only supported runtime is direct Rust access to `LibTorch` through tch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BackendSelection {
    /// Retained as a configuration spelling during migration; it resolves to
    /// the sole tch/LibTorch runtime.
    #[default]
    Auto,
    /// Explicitly select tch/LibTorch.
    LibTorch,
}

impl BackendSelection {
    /// Parse the stable configuration spelling.
    ///
    /// # Errors
    ///
    /// Returns an error when a caller supplies a backend name other than the
    /// accepted compatibility spellings.
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            None | Some("auto" | "libtorch" | "torchscript" | "tch") => Ok(Self::LibTorch),
            Some(other) => {
                eyre::bail!("unknown backend {other:?}; teamy-tts supports only tch/LibTorch")
            }
        }
    }

    /// Return the stable CLI/configuration spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto | Self::LibTorch => "libtorch",
        }
    }
}

impl fmt::Display for BackendSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
