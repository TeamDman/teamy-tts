resource "cloudflare_r2_bucket" "models" {
  account_id = "fcdcf78a5f4ff76266c9c6cfb664d01d"
  name       = "teamy-tts-models"

  # Cloudflare has no Canada-specific R2 hint. ENAM is the closest supported
  # hint for Ottawa and Eastern Canada; placement is best effort.
  location = "ENAM"
}
