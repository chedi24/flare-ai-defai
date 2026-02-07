import structlog
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)

class Settings(BaseSettings):
    # Flags
    simulate_attestation: bool = False
    simulate_ai: bool = False

    # CORS
    cors_origins: list[str] = ["*"]

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # API
    api_version: str = "v1"

    # Web3
    web3_provider_url: str = "https://coston2-api.flare.network/ext/C/rpc"
    web3_explorer_url: str = "https://coston2-explorer.flare.network/"

    # Snapshot JSON (dummy now, real later)
    latest_update_path: str = "shared/latest_update.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
logger.debug("settings", settings=settings.model_dump())
