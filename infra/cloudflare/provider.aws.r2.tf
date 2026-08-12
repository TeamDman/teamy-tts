# R2 exposes an S3-compatible API. The AWS provider is only the Terraform
# client for that Cloudflare API; it does not create AWS storage. The region is
# a signing placeholder for the S3 client; R2 placement is set to ENAM on the
# Cloudflare bucket resource.
provider "aws" {
  alias  = "r2"
  region = "us-east-1"

  skip_credentials_validation = true
  skip_region_validation      = true
  skip_requesting_account_id  = true
  skip_metadata_api_check     = true

  endpoints {
    s3 = "https://fcdcf78a5f4ff76266c9c6cfb664d01d.r2.cloudflarestorage.com"
  }
}
