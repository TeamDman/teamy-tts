# Retain this release when adding future archives; a key change on this resource
# would otherwise delete the old object.
resource "aws_s3_object" "models_archive" {
  provider = aws.r2

  bucket        = cloudflare_r2_bucket.models.name
  key           = local.models_object_key
  source        = local.models_archive_source
  source_hash   = local.models_archive_sha256
  content_type  = "application/zip"
  cache_control = "public, max-age=31536000, immutable"

  metadata = {
    sha256 = local.models_archive_sha256
  }

  lifecycle {
    prevent_destroy = true
  }
}
