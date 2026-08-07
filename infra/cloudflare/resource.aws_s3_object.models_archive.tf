# The object key is content-addressed, so a changed archive creates a new
# immutable object instead of silently replacing the old one.
resource "aws_s3_object" "models_archive" {
  provider = aws.r2

  bucket        = cloudflare_r2_bucket.models.name
  key           = local.models_object_key
  source        = "../../models.zip"
  source_hash   = local.models_archive_sha256
  content_type  = "application/zip"
  cache_control = "public, max-age=31536000, immutable"

  metadata = {
    sha256 = local.models_archive_sha256
  }
}
