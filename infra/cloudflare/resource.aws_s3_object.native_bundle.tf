# The hash and upload refer to the same file. Retain old release resources when
# publishing future versions: changing one resource's key would delete its old URL.
resource "aws_s3_object" "native_bundle_tch" {
  provider = aws.r2

  bucket        = cloudflare_r2_bucket.models.name
  key           = local.native_bundle_object_key
  source        = local.native_bundle_archive_source
  source_hash   = local.native_bundle_archive_sha256
  content_type  = "application/zip"
  cache_control = "public, max-age=31536000, immutable"

  metadata = {
    sha256 = local.native_bundle_archive_sha256
  }

  lifecycle {
    prevent_destroy = true
  }
}
