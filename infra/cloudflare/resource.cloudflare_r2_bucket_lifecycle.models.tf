# Keep abandoned multipart uploads from accumulating storage charges. Use
# Cloudflare's native R2 API because the S3-compatible lifecycle call can hang
# against R2 even though object upload succeeds.
resource "cloudflare_r2_bucket_lifecycle" "models" {
  account_id  = var.cloudflare_account_id
  bucket_name = "teamy-tts-models"

  rules = [{
    id = "abort-incomplete-multipart-uploads"
    conditions = {
      prefix = ""
    }
    enabled = true
    abort_multipart_uploads_transition = {
      condition = {
        max_age = 604800
        type    = "Age"
      }
    }
  }]
}
