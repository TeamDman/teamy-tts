variable "cloudflare_account_id" {
  type        = string
  description = "Cloudflare account ID, supplied through TF_VAR_cloudflare_account_id."

  validation {
    condition     = can(regex("^[a-f0-9]{32}$", var.cloudflare_account_id))
    error_message = "cloudflare_account_id must be a 32-character lowercase hexadecimal account ID."
  }
}
