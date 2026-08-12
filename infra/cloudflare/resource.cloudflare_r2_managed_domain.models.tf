resource "cloudflare_r2_managed_domain" "models" {
  account_id  = "fcdcf78a5f4ff76266c9c6cfb664d01d"
  bucket_name = cloudflare_r2_bucket.models.name
  enabled     = true
}
