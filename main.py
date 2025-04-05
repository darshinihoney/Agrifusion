import os
import argparse
import sqlite3
from agents.farmer_agnet import FarmerIntelligenceAgent
from agents.market_agent import MarketDynamicsAgent
from models.predictor import WeatherPredictor
from models.forecasting import MarketForecaster
from utils.preprocessing import DataPreprocessor
from visualizations.insights_plot import InsightsVisualizer
from APIs.fastapi_server import start_api_server

def setup_database():
    """Initialize SQLite database for long-term memory storage"""
    print("Setting up SQLite database for long-term memory...")
    conn = sqlite3.connect('agrifusion.db')
    cursor = conn.cursor()
    
    # Create tables for storing agent insights and recommendations
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS farmer_recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farm_id TEXT,
        recommendation TEXT,
        crop_suggestion TEXT,
        planting_date TEXT,
        expected_yield REAL,
        water_usage_estimate REAL,
        carbon_footprint_estimate REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_insights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        crop TEXT,
        current_price REAL,
        forecasted_price REAL,
        demand_trend TEXT,
        region TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS farm_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farm_id TEXT,
        soil_type TEXT,
        location TEXT, 
        size_hectares REAL,
        water_source TEXT,
        previous_crops TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Database setup complete!")

def main():
    """Main entry point for the Dual Synapse Agrifusion system"""
    parser = argparse.ArgumentParser(description='Dual Synapse Agrifusion - Multi-agent agricultural AI system')
    parser.add_argument('--mode', choices=['api', 'cli'], default='cli', help='Run in API server mode or CLI mode')
    parser.add_argument('--farm_id', type=str, help='Farm ID for recommendations')
    parser.add_argument('--region', type=str, help='Region for market analysis')
    args = parser.parse_args()
    
    # Initialize database if it doesn't exist
    if not os.path.exists('agrifusion.db'):
        setup_database()
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess datasets
    farmer_data = preprocessor.load_and_process('datasets/farmer_advisor_dataset.csv')
    market_data = preprocessor.load_and_process('datasets/market_researcher_dataset.csv')
    
    # Initialize models
    weather_predictor = WeatherPredictor()
    market_forecaster = MarketForecaster()
    
    # Initialize agents
    farmer_agent = FarmerIntelligenceAgent(weather_predictor, farmer_data)
    market_agent = MarketDynamicsAgent(market_forecaster, market_data)
    
    # Initialize visualizer
    visualizer = InsightsVisualizer()
    
    if args.mode == 'api':
        # Start FastAPI server
        start_api_server(farmer_agent, market_agent, visualizer)
    else:
        # CLI mode
        if args.farm_id:
            farm_recommendations = farmer_agent.generate_recommendations(args.farm_id)
            print("Farmer Intelligence Agent Recommendations:")
            print(farm_recommendations)
            
        if args.region:
            market_insights = market_agent.analyze_market(args.region)
            print("\nMarket Dynamics Agent Insights:")
            print(market_insights)
            
            # Generate and display visualizations
            visualizer.plot_market_trends(market_insights)
            
        if not args.farm_id and not args.region:
            print("Please provide either --farm_id or --region arguments for recommendations")
            print("Example: python main.py --farm_id FARM123 --region California")

if __name__ == "__main__":
    main()