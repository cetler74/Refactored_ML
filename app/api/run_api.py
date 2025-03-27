import logging
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_server.log"),
    ],
)

logger = logging.getLogger(__name__)

from app.core.orchestrator import BotOrchestrator
from app.config.settings import Settings
from app.api.server import APIServer

# Create settings and orchestrator instances
settings = Settings()
orchestrator = BotOrchestrator(settings)

# Create API server instance
api_server = APIServer(settings, orchestrator)

# Export the FastAPI app for uvicorn
app = api_server.app 