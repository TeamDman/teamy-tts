output "teamy_raw_source_url" {
  description = "Public URL to the immutable raw GLaDOS model archive; bake this into the Teamy source default."
  value       = "https://${cloudflare_r2_managed_domain.models.domain}/${local.models_object_key}"
}
