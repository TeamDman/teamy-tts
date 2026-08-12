# The object key is content-addressed, so a changed native bundle creates an
# immutable object instead of silently replacing the prepared model source.
resource "aws_s3_object" "native_bundle" {
  provider = aws.r2

  bucket        = cloudflare_r2_bucket.models.name
  key           = local.native_bundle_object_key
  source        = "../../artifacts/teamy-tts-glados-new-tch-native-bundle.zip"
  source_hash   = local.native_bundle_archive_sha256
  content_type  = "application/zip"
  cache_control = "public, max-age=31536000, immutable"

  metadata = {
    sha256 = local.native_bundle_archive_sha256
  }
}
