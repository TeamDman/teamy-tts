use crate::backend::BackendKind;
use crate::backend_receipts;
use crate::cli::output::CliOutput;
use crate::model_registry;
use arbitrary::Arbitrary;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use figue as args;

/// List concrete backend candidates and the current automatic-selection state.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct BackendListArgs {
    /// Stable model identifier. Defaults to glados.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub model: Option<String>,
}

#[derive(Clone, Debug, Facet, PartialEq)]
struct BackendCandidateReport {
    backend: String,
    available: bool,
    selected_by_auto: bool,
    status: String,
}

#[derive(Clone, Debug, Facet, PartialEq)]
struct BackendListReport {
    model: String,
    device_identity: String,
    automatic_backend: String,
    automatic_reason: String,
    automatic_receipt_path: Option<String>,
    candidates: Vec<BackendCandidateReport>,
}

impl BackendListArgs {
    /// # Errors
    ///
    /// Returns an error when the requested model is unknown.
    #[expect(
        clippy::unused_async,
        reason = "Command invoke methods share the async CLI dispatch shape."
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let model_id = self.model.as_deref().unwrap_or("glados");
        let Some(model) = model_registry::find_model(model_id) else {
            bail!("unknown model {model_id:?}; known models: glados");
        };

        let prepared = model_registry::inspect_prepared_model_dir(model).ok();
        let device_identity = backend_receipts::device_identity();
        let configuration = backend_receipts::BenchmarkConfiguration::default();
        let decision = prepared.as_ref().map_or_else(
            || backend_receipts::AutoBackendDecision {
                backend: BackendKind::Burn.to_string(),
                reason: "prepared model is not verified; Burn is the documented fallback"
                    .to_string(),
                receipt_path: None,
            },
            |prepared| {
                backend_receipts::auto_backend_decision(prepared, &device_identity, &configuration)
            },
        );

        let candidates = candidate_reports(prepared.is_some(), &decision.backend);

        Ok(CliOutput::facet(BackendListReport {
            model: model.id.to_string(),
            device_identity,
            automatic_backend: decision.backend,
            automatic_reason: decision.reason,
            automatic_receipt_path: decision.receipt_path,
            candidates,
        }))
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "The backend list keeps each candidate's availability and explanation adjacent."
)]
fn candidate_reports(prepared: bool, automatic_backend: &str) -> Vec<BackendCandidateReport> {
    [
            (
                BackendKind::Burn,
                prepared,
                if prepared {
                    "Burn CPU-acoustic/CUDA-vocoder split is available"
                } else {
                    "prepared model is not verified"
                },
            ),
            (
                BackendKind::BurnNdArray,
                prepared,
                if prepared {
                    "both neural graphs use the portable Burn NdArray backend"
                } else {
                    "prepared model is not verified"
                },
            ),
            (
                BackendKind::BurnCudaAcoustic,
                prepared && backend_receipts::backend_is_available("burn-cuda-acoustic"),
                if !prepared {
                    "prepared model is not verified"
                } else if backend_receipts::backend_is_available("burn-cuda-acoustic") {
                    "both neural graphs use Burn CUDA"
                } else {
                    "CUDA feature is unavailable in this build"
                },
            ),
            (
                BackendKind::BurnCudaFused,
                prepared && backend_receipts::backend_is_available("burn-cuda-fused"),
                if !prepared {
                    "prepared model is not verified"
                } else if backend_receipts::backend_is_available("burn-cuda-fused") {
                    "Burn CUDA fusion/autotune build is available"
                } else {
                    "build with --features burn-cuda-fused to enable this candidate"
                },
            ),
            (
                BackendKind::BurnTch,
                prepared && backend_receipts::backend_is_available("burn-tch"),
                if !prepared {
                    "prepared model is not verified"
                } else if backend_receipts::backend_is_available("burn-tch") {
                    "Burn native model runs through the LibTorch/tch backend"
                } else {
                    "build with --features burn-tch to enable this candidate"
                },
            ),
            (
                BackendKind::BurnWgpu,
                prepared && backend_receipts::backend_is_available("burn-wgpu"),
                if !prepared {
                    "prepared model is not verified"
                } else if backend_receipts::backend_is_available("burn-wgpu") {
                    "Burn WGPU uses automatic graphics API selection"
                } else {
                    "build with --features burn-wgpu to enable this candidate"
                },
            ),
            (
                BackendKind::BurnVulkan,
                prepared && backend_receipts::backend_is_available("burn-vulkan"),
                if !prepared {
                    "prepared model is not verified"
                } else if backend_receipts::backend_is_available("burn-vulkan") {
                    "Burn uses explicit Vulkan graphics and SPIR-V kernels"
                } else {
                    "build with --features burn-vulkan"
                },
            ),
            (
                BackendKind::LibTorch,
                prepared && backend_receipts::backend_is_available("libtorch"),
                if !prepared {
                    "prepared model is not verified"
                } else if backend_receipts::backend_is_available("libtorch") {
                    "TorchScript feature and model directory are available"
                } else {
                    "TorchScript feature or model directory is unavailable"
                },
            ),
            (
                BackendKind::Vulkan,
                backend_receipts::backend_is_available("vulkan"),
                if backend_receipts::backend_is_available("vulkan") {
                    "Vulkan acoustic embedding/condition projections/LSTM/mel/postnet/post projection and HiFi-GAN vocoder are available; predictor/prenet remain Burn-backed"
                } else {
                    "Vulkan feature is unavailable in this build"
                },
            ),
        ]
        .into_iter()
        .map(|(backend, available, status)| BackendCandidateReport {
            backend: backend.to_string(),
            available,
            selected_by_auto: automatic_backend == backend.to_string(),
            status: status.to_string(),
        })
        .collect()
}
