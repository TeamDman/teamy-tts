# Preserve the originally published .bpk bundle. A later configuration edit
# switched the upload source to .pt without switching the hash source. Keep the
# historical bytes and publish the .pt artifact under its own correct hash.
moved {
  from = aws_s3_object.native_bundle
  to   = aws_s3_object.native_bundle_historical
}

resource "aws_s3_object" "native_bundle_historical" {
  provider = aws.r2
  bucket   = cloudflare_r2_bucket.models.name
  key      = "native/glados/ab663a68fb5263b8df49f76b80812ba2692b5d1a0234a246528d65d89fd2f81f/native-bundle.zip"

  lifecycle {
    prevent_destroy = true
    ignore_changes  = all
  }
}
