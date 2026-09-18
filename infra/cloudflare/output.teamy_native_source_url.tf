output "teamy_native_source_url" {
  description = "Public URL to the immutable native GLaDOS bundle; bake this into the Teamy source default."
  value       = "https://${cloudflare_r2_managed_domain.models.domain}/${aws_s3_object.native_bundle_tch.key}"
}
