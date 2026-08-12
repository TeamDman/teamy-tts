terraform {
  required_version = ">= 1.6.0"

  backend "azurerm" {
    subscription_id      = "6cb7032f-2437-4f5e-91e8-676cb67e5444"
    resource_group_name  = "Teamy-Infrastructure-as-Code-RG"
    storage_account_name = "teamyiac"
    container_name       = "statefiles"
    key                  = "teamy-tts/cloudflare.tfstate"
  }

  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5.0"
    }

    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
