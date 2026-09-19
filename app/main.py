import os
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from app.api.routes import router as api_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("isl_translation")


from app.stt.whisper_service import init_stt_provider

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Speech -> Sign Language Pipeline API server...")
    try:
        init_stt_provider()
    except Exception as e:
        logger.error(f"Error initializing STT Provider during startup: {e}")
    yield


app = FastAPI(
    title="Speech & Text -> Sign Language (SiGML / CWASA) Pipeline API",
    description="Backend API and testing dashboard for converting Speech and Text to CWASA-ready SiGML XML",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend / hackathon client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router)

# Mount Static Web UI Dashboard
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


