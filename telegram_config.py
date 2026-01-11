# Telegram Bot Configuration
# ==========================
# Uses .env file for security (tokens not in source code)
#
# Setup:
# 1. Create .env file with: TELEGRAM_BOT_TOKEN=your_token_here
# 2. Add TELEGRAM_CHAT_ID=your_chat_id after running --get-chat-id

import os
from pathlib import Path

# Try to load from .env file
_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _ROOT / ".env"

def _load_env():
    """Load environment variables from .env file."""
    if _ENV_FILE.exists():
        with open(_ENV_FILE, encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

_load_env()

# Your Telegram Bot Token (from BotFather)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Your Chat ID (auto-detected or set manually)
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Notification settings
TELEGRAM_TOP_N_PICKS = 10  # How many top picks to include
TELEGRAM_MIN_EV = 0.05     # Minimum EV threshold (5%)

# Weekly/Daily automation settings
TELEGRAM_MIN_ODDS = 1.60
TELEGRAM_WEEKLY_HORIZON_DAYS = 7
TELEGRAM_TZ = "Europe/Bucharest"
TELEGRAM_MESSAGE_MAX_CHARS = 3500
