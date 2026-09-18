variable "models_archive_source" {
  description = "Optional local path to the original raw model ZIP."
  type        = string
  default     = null
}

variable "native_bundle_archive_source" {
  description = "Optional local path to the legacy TorchScript runtime ZIP."
  type        = string
  default     = null
}

variable "cuda_bundle_archive_source" {
  description = "Optional local path to the deterministic safetensors runtime ZIP."
  type        = string
  default     = null
}

locals {
  models_archive_source        = coalesce(var.models_archive_source, "../../models.zip")
  native_bundle_archive_source = coalesce(var.native_bundle_archive_source, "../../artifacts/teamy-tts-glados-new-tch-native-bundle.zip")
  cuda_bundle_archive_source   = coalesce(var.cuda_bundle_archive_source, "../../artifacts/teamy-tts-glados-native-v1.zip")
}
