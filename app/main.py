"""FastAPI application entry point"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.config import settings
from src.app import __version__

# Create FastAPI application
app = FastAPI(
    title="City Permitting Knowledge Graph API",
    description="A structured, machine-readable knowledge graph system for municipal permitting rules",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "City Permitting Knowledge Graph API",
        "version": __version__,
        "status": "operational",
        "environment": settings.environment
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "openai_configured": bool(settings.openai_api_key)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )
