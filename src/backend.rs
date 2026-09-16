//! Select the inference backend compiled into this executable.

use eyre::Result;
use std::fmt;

/// Stable backend names shared by the default and CUDA-native builds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BackendSelection {
    /// Resolve to the backend enabled at build time.
    #[default]
    Auto,
    /// Explicitly select tch/LibTorch.
    LibTorch,
    /// Source-defined CUDA model without LibTorch.
    NativeCuda,
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
            None | Some("auto") => Ok(if cfg!(feature = "cuda-native") {
                Self::NativeCuda
            } else {
                Self::LibTorch
            }),
            Some("native" | "cuda-native") if cfg!(feature = "cuda-native") => Ok(Self::NativeCuda),
            Some("libtorch" | "torchscript" | "tch") if cfg!(feature = "tch-native") => {
                Ok(Self::LibTorch)
            }
            Some(other) => {
                eyre::bail!("backend {other:?} is unavailable in this build")
            }
        }
    }

    /// Return the stable CLI/configuration spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto if cfg!(feature = "cuda-native") => "cuda-native",
            Self::Auto | Self::LibTorch => "libtorch",
            Self::NativeCuda => "cuda-native",
        }
    }
}

impl fmt::Display for BackendSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
