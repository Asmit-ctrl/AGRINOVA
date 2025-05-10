from fastapi import FastAPI
from crop_recommendation.api import npk, crops, farmers
from crop_recommendation.db.database import Base, engine

# Initialize database
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(title="Crop Recommendation System")

# Include routers
app.include_router(npk.router, prefix="/api/npk", tags=["NPK Estimation"])
app.include_router(crops.router, prefix="/api/crops", tags=["Crops"])
app.include_router(farmers.router, prefix="/api/farmers", tags=["Farmers"])
