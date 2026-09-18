locals {
  cuda_bundle_catalog = jsondecode(file("${path.module}/../../assets/glados-native-v1.json"))
  cuda_bundle_sha256  = filesha256(local.cuda_bundle_archive_source)
  cuda_bundle_key     = "cuda-native/glados/v1/${local.cuda_bundle_sha256}/model.zip"
}

resource "aws_s3_object" "cuda_bundle_v1" {
  provider      = aws.r2
  bucket        = cloudflare_r2_bucket.models.name
  key           = local.cuda_bundle_key
  source        = local.cuda_bundle_archive_source
  source_hash   = local.cuda_bundle_sha256
  content_type  = "application/zip"
  cache_control = "public, max-age=31536000, immutable"
  metadata = {
    sha256 = local.cuda_bundle_sha256
    format = "teamy-glados-native-v1"
  }

  lifecycle {
    prevent_destroy = true
    precondition {
      condition     = local.cuda_bundle_sha256 == local.cuda_bundle_catalog.archive_sha256
      error_message = "CUDA archive hash must match the pinned application catalog."
    }
  }
}

output "teamy_cuda_source_url" {
  value = "https://${cloudflare_r2_managed_domain.models.domain}/${aws_s3_object.cuda_bundle_v1.key}"
}
