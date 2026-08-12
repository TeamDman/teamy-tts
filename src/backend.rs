//! Workload-level inference backend contracts.
//!
//! Backends deliberately receive model-facing values rather than Burn tensors.
//! Each implementation owns its tensor types, device buffers, layouts, and
//! reusable workspaces so accelerated candidates can remain native to their
//! runtime.

use eyre::Result;
use std::fmt;

/// A backend that can synthesize one prepared `GLaDOS` utterance.
pub(crate) trait GladosBackend: fmt::Debug {
    /// Identify the concrete runtime used by this backend.
    fn kind(&self) -> BackendKind;

    /// Synthesize mono floating-point audio for one prepared utterance.
    fn synthesize(&self, input: &SynthesisInput<'_>) -> Result<Vec<f32>>;
}

/// Inputs shared by all `GLaDOS` inference candidates.
#[derive(Debug)]
pub(crate) struct SynthesisInput<'a> {
    /// Frontend-produced token IDs in model order.
    pub tokens: &'a [i32],
    /// Prepared speaker identity, such as `p1` or `p2`.
    pub voice: &'a str,
    /// Upstream duration/pitch scaling factor.
    pub alpha: f32,
}

/// Inference backend candidates known to teamy-tts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// Burn with the current production split: CPU acoustic graph and CUDA
    /// vocoder when CUDA is compiled in.
    Burn,
    /// Burn with both neural graphs on the portable `NdArray` backend.
    BurnNdArray,
    /// Burn with both neural graphs on CUDA.
    BurnCudaAcoustic,
    /// Burn CUDA with Burn's fusion/autotune build features enabled.
    BurnCudaFused,
    /// Burn's LibTorch/tch backend executing the native Burn model.
    BurnTch,
    /// Burn WGPU using its automatic graphics API selection.
    BurnWgpu,
    /// Burn's explicit Vulkan backend using CubeCL's Vulkan/SPIR-V path.
    BurnVulkan,
    /// Upstream `TorchScript` executed through native `LibTorch`.
    LibTorch,
    /// Specialized Vulkan compute implementation.
    Vulkan,
}

impl BackendKind {
    /// Parse a stable backend identity spelling.
    ///
    /// # Errors
    ///
    /// Returns an error when the value is not a concrete backend identity.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "burn" => Ok(Self::Burn),
            "burn-ndarray" | "burn-cpu" => Ok(Self::BurnNdArray),
            "burn-cuda-acoustic" | "burn-cuda" => Ok(Self::BurnCudaAcoustic),
            "burn-cuda-fused" | "burn-fused" => Ok(Self::BurnCudaFused),
            "burn-tch" | "burn-libtorch" => Ok(Self::BurnTch),
            "burn-wgpu" => Ok(Self::BurnWgpu),
            "burn-vulkan" => Ok(Self::BurnVulkan),
            "libtorch" | "torchscript" => Ok(Self::LibTorch),
            "vulkan" => Ok(Self::Vulkan),
            other => eyre::bail!("unknown backend identity {other:?}"),
        }
    }

    /// Return the stable CLI/configuration spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Burn => "burn",
            Self::BurnNdArray => "burn-ndarray",
            Self::BurnCudaAcoustic => "burn-cuda-acoustic",
            Self::BurnCudaFused => "burn-cuda-fused",
            Self::BurnTch => "burn-tch",
            Self::BurnWgpu => "burn-wgpu",
            Self::BurnVulkan => "burn-vulkan",
            Self::LibTorch => "libtorch",
            Self::Vulkan => "vulkan",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// User-selected backend policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BackendSelection {
    /// Prefer an available accelerated candidate, then use Burn.
    #[default]
    Auto,
    /// Force the current Burn CPU-acoustic/CUDA-vocoder candidate.
    Burn,
    /// Force Burn with both neural graphs on `NdArray`.
    BurnNdArray,
    /// Force Burn with both neural graphs on CUDA.
    BurnCudaAcoustic,
    /// Force the fused/autotuned Burn CUDA build candidate.
    BurnCudaFused,
    /// Force Burn's LibTorch/tch backend.
    BurnTch,
    /// Force Burn WGPU using its automatic graphics API selection.
    BurnWgpu,
    /// Force Burn's explicit Vulkan backend.
    BurnVulkan,
    /// Force the native LibTorch/TorchScript backend.
    LibTorch,
    /// Force the Vulkan backend.
    Vulkan,
}

impl BackendSelection {
    /// Parse the optional CLI value, defaulting to automatic selection.
    ///
    /// # Errors
    ///
    /// Returns an error when the value is not one of the supported backend
    /// spellings.
    pub fn parse(value: Option<&str>) -> Result<Self> {
        let Some(value) = value else {
            return Ok(Self::Auto);
        };

        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "burn" => Ok(Self::Burn),
            "burn-ndarray" | "burn-cpu" => Ok(Self::BurnNdArray),
            "burn-cuda-acoustic" | "burn-cuda" => Ok(Self::BurnCudaAcoustic),
            "burn-cuda-fused" | "burn-fused" => Ok(Self::BurnCudaFused),
            "burn-tch" | "burn-libtorch" => Ok(Self::BurnTch),
            "burn-wgpu" => Ok(Self::BurnWgpu),
            "burn-vulkan" => Ok(Self::BurnVulkan),
            "libtorch" | "torchscript" => Ok(Self::LibTorch),
            "vulkan" => Ok(Self::Vulkan),
            other => {
                eyre::bail!(
                    "unknown backend {other:?}; expected auto, burn, burn-ndarray, burn-cuda-acoustic, burn-cuda-fused, burn-tch, burn-wgpu, burn-vulkan, libtorch, or vulkan"
                )
            }
        }
    }

    /// Return the stable CLI/configuration spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Burn => "burn",
            Self::BurnNdArray => "burn-ndarray",
            Self::BurnCudaAcoustic => "burn-cuda-acoustic",
            Self::BurnCudaFused => "burn-cuda-fused",
            Self::BurnTch => "burn-tch",
            Self::BurnWgpu => "burn-wgpu",
            Self::BurnVulkan => "burn-vulkan",
            Self::LibTorch => "libtorch",
            Self::Vulkan => "vulkan",
        }
    }
}

impl fmt::Display for BackendSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::BackendKind;
    use super::BackendSelection;

    #[test]
    fn backend_selection_defaults_to_auto() {
        assert_eq!(
            BackendSelection::parse(None).unwrap(),
            BackendSelection::Auto
        );
    }

    #[test]
    fn backend_selection_accepts_stable_spellings() {
        assert_eq!(
            BackendSelection::parse(Some("burn")).unwrap(),
            BackendSelection::Burn
        );
        assert_eq!(
            BackendSelection::parse(Some("TorchScript")).unwrap(),
            BackendSelection::LibTorch
        );
        assert_eq!(
            BackendSelection::parse(Some("burn-cpu")).unwrap(),
            BackendSelection::BurnNdArray
        );
        assert_eq!(
            BackendSelection::parse(Some("burn-cuda")).unwrap(),
            BackendSelection::BurnCudaAcoustic
        );
        assert_eq!(
            BackendSelection::parse(Some("burn-fused")).unwrap(),
            BackendSelection::BurnCudaFused
        );
        assert_eq!(
            BackendSelection::parse(Some("burn-libtorch")).unwrap(),
            BackendSelection::BurnTch
        );
        assert_eq!(
            BackendSelection::parse(Some("burn-wgpu")).unwrap(),
            BackendSelection::BurnWgpu
        );
        assert_eq!(
            BackendSelection::parse(Some("burn-vulkan")).unwrap(),
            BackendSelection::BurnVulkan
        );
        assert_eq!(
            BackendSelection::parse(Some("VULKAN")).unwrap(),
            BackendSelection::Vulkan
        );
    }

    #[test]
    fn backend_selection_rejects_unknown_values() {
        let error = BackendSelection::parse(Some("cuda")).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("expected auto, burn, burn-ndarray")
        );
    }

    #[test]
    fn backend_kind_spellings_are_stable() {
        assert_eq!(BackendKind::Burn.to_string(), "burn");
        assert_eq!(BackendKind::BurnNdArray.to_string(), "burn-ndarray");
        assert_eq!(
            BackendKind::BurnCudaAcoustic.to_string(),
            "burn-cuda-acoustic"
        );
        assert_eq!(BackendKind::BurnCudaFused.to_string(), "burn-cuda-fused");
        assert_eq!(BackendKind::BurnTch.to_string(), "burn-tch");
        assert_eq!(BackendKind::BurnWgpu.to_string(), "burn-wgpu");
        assert_eq!(BackendKind::BurnVulkan.to_string(), "burn-vulkan");
        assert_eq!(BackendKind::LibTorch.to_string(), "libtorch");
        assert_eq!(BackendKind::Vulkan.to_string(), "vulkan");
    }
}
