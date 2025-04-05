from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sqlite3
import uvicorn
import json
import os
from datetime import datetime

app = FastAPI(title="Dual Synapse Agrifusion API", 
              description="API for the Dual Synapse Agrifusion multi-agent agricultural AI system",
              version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store agent instances
farmer_agent = None
market_agent = None
visualizer = None

# Pydantic models for request/response validation
class FarmData(BaseModel):
    farm_id: str
    soil_type: str
    location: str
    size_hectares: float
    water_source: str
    previous_crops: str

class Recommendation(BaseModel):
    farm_id: str
    recommendation: str
    crop_suggestion: str
    planting_date: str
    expected_yield: float
    water_usage_estimate: float
    carbon_footprint_estimate: float
    timestamp: Optional[str] = None

class MarketInsight(BaseModel):
    crop: str
    current_price: float
    forecasted_price: float
    demand_trend: str
    region: str
    timestamp: Optional[str] = None

class MarketAnalysis(BaseModel):
    region: str
    top_recommendations: List[Dict[str, Any]]
    market_overview: Dict[str, Any]
    previous_insights: List[Dict[str, Any]]

def get_db_connection():
    """Create a database connection and return it"""
    conn = sqlite3.connect('agrifusion.db')
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Welcome to the Dual Synapse Agrifusion API"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/farms", response_model=Dict[str, str])
def register_farm(farm_data: FarmData):
    """Register a new farm in the system"""
    if farmer_agent is None:
        raise HTTPException(status_code=503, detail="Farmer agent not initialized")
    
    try:
        farmer_agent.store_farm_data(farm_data.dict())
        return {"message": f"Farm {farm_data.farm_id} registered successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register farm: {str(e)}")

@app.get("/farms/{farm_id}", response_model=FarmData)
def get_farm(farm_id: str):
    """Retrieve farm information"""
    if farmer_agent is None:
        raise HTTPException(status_code=503, detail="Farmer agent not initialized")
    
    farm_data = farmer_agent.get_farm_data(farm_id)
    if farm_data is None:
        raise HTTPException(status_code=404, detail=f"Farm {farm_id} not found")
    
    return farm_data

@app.get("/recommendations/{farm_id}", response_model=Recommendation)
def get_recommendations(farm_id: str):
    """Generate farming recommendations for a specific farm"""
    if farmer_agent is None:
        raise HTTPException(status_code=503, detail="Farmer agent not initialized")
    
    try:
        recommendations = farmer_agent.generate_recommendations(farm_id)
        if "error" in recommendations:
            raise HTTPException(status_code=404, detail=recommendations["error"])
        
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@app.get("/market/{region}", response_model=MarketAnalysis)
def get_market_analysis(region: str):
    """Analyze market conditions for a specific region"""
    if market_agent is None:
        raise HTTPException(status_code=503, detail="Market agent not initialized")
    
    try:
        analysis = market_agent.analyze_market(region)
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze market: {str(e)}")

@app.get("/market/history/{region}")
def get_market_history(region: str, limit: int = Query(10, gt=0, le=100)):
    """Get historical market insights for a region"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT crop, current_price, forecasted_price, demand_trend, timestamp
        FROM market_insights
        WHERE region = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """, (region, limit))
        
        results = cursor.fetchall()
        insights = []
        
        for row in results:
            insights.append(dict(row))
        
        return {"region": region, "insights": insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve market history: {str(e)}")
    finally:
        conn.close()

@app.get("/recommendations/history/{farm_id}")
def get_recommendation_history(farm_id: str, limit: int = Query(10, gt=0, le=100)):
    """Get historical recommendations for a farm"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT recommendation, crop_suggestion, planting_date, expected_yield, 
               water_usage_estimate, carbon_footprint_estimate, timestamp
        FROM farmer_recommendations
        WHERE farm_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """, (farm_id, limit))
        
        results = cursor.fetchall()
        recommendations = []
        
        for row in results:
            recommendations.append(dict(row))
        
        return {"farm_id": farm_id, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve recommendation history: {str(e)}")
    finally:
        conn.close()

def start_api_server(farmer_intelligence_agent, market_dynamics_agent, insights_visualizer, host="0.0.0.0", port=8000):
    """Start the FastAPI server"""
    global farmer_agent, market_agent, visualizer
    
    # Store agent instances in global variables
    farmer_agent = farmer_intelligence_agent
    market_agent = market_dynamics_agent
    visualizer = insights_visualizer
    
    # Start server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    print("This module should be imported and used via the main.py file")
    print("To start the server directly (not recommended), use:")
    print("uvicorn api.fastapi_server:app --reload")