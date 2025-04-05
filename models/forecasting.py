import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import logging
import sqlite3
from datetime import datetime, timedelta

class MarketForecaster:
    """
    The MarketForecaster class provides crop price and demand forecasting
    based on historical market data, seasonal patterns, and external factors.
    """
    
    def __init__(self, data_source=None):
        """
        Initialize the market forecaster model.
        
        Args:
            data_source (pd.DataFrame, optional): Initial market data to train the model.
                                                If None, data will be loaded from the database.
        """
        self.logger = logging.getLogger(__name__)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
        if data_source is not None:
            self.train(data_source)
        else:
            self.logger.info("No initial data provided. Please load or train data.")
    
    def _connect_to_db(self):
        """Create a connection to the SQLite database"""
        return sqlite3.connect('agrifusion.db')
    
    def train(self, market_data):
        """
        Train the forecasting model with historical market data.
        
        Args:
            market_data (pd.DataFrame): Historical market data with features and target variables
        
        Returns:
            dict: Training metrics including MAE and R2 score
        """
        self.logger.info("Training market forecasting model...")
        
        # Check if data is valid
        if market_data is None or len(market_data) < 10:
            self.logger.warning("Insufficient data for training. Need at least 10 records.")
            return {"status": "error", "message": "Insufficient training data"}
        
        try:
            # Prepare features and target
            X = market_data.drop(['crop_price', 'crop_name'], axis=1, errors='ignore')
            y = market_data['crop_price']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.is_trained = True
            self.feature_cols = X.columns.tolist()
            
            self.logger.info(f"Model trained successfully. MAE: {mae:.2f}, R2: {r2:.2f}")
            
            # Store training metrics in the database
            self._save_model_metrics(mae, r2)
            
            return {
                "status": "success",
                "metrics": {
                    "mae": mae,
                    "r2": r2
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _save_model_metrics(self, mae, r2):
        """Save model training metrics to the database"""
        try:
            conn = self._connect_to_db()
            cursor = conn.cursor()
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                mae REAL,
                r2 REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            cursor.execute(
                "INSERT INTO model_metrics (model_name, mae, r2) VALUES (?, ?, ?)",
                ("market_forecaster", mae, r2)
            )
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error saving model metrics: {str(e)}")
    
    def forecast_prices(self, crop, region, forecast_period=30):
        """
        Forecast crop prices for the specified period.
        
        Args:
            crop (str): Crop name
            region (str): Target region
            forecast_period (int): Number of days to forecast
        
        Returns:
            dict: Forecast results including daily prices and trends
        """
        self.logger.info(f"Forecasting prices for {crop} in {region} for the next {forecast_period} days")
        
        if not self.is_trained:
            self.logger.warning("Model not trained. Please train the model first.")
            return {"status": "error", "message": "Model not trained"}
        
        try:
            # Get historical data from database
            conn = self._connect_to_db()
            historic_data = pd.read_sql(
                f"SELECT * FROM market_insights WHERE crop = '{crop}' AND region = '{region}' ORDER BY timestamp DESC LIMIT 90",
                conn
            )
            conn.close()
            
            if len(historic_data) < 5:
                self.logger.warning(f"Insufficient historical data for {crop} in {region}")
                # Use backup estimation method
                return self._estimate_prices_without_model(crop, region, forecast_period)
            
            # Prepare forecast data
            today = datetime.now()
            forecast_dates = [today + timedelta(days=i) for i in range(forecast_period)]
            
            # Generate features for each forecast date
            forecast_features = []
            for date in forecast_dates:
                # Create feature row based on date and seasonality
                features = {
                    'month': date.month,
                    'day_of_year': date.timetuple().tm_yday,
                    'season': (date.month % 12 + 3) // 3,  # 1: spring, 2: summer, 3: fall, 4: winter
                }
                
                # Add region and crop encoding
                # In a real system, you would use proper encoding
                features['region_code'] = hash(region) % 100  
                features['crop_code'] = hash(crop) % 100
                
                forecast_features.append(features)
            
            forecast_df = pd.DataFrame(forecast_features)
            
            # Ensure forecast_df has the same columns as training data
            for col in self.feature_cols:
                if col not in forecast_df.columns:
                    forecast_df[col] = 0
            
            forecast_df = forecast_df[self.feature_cols]
            
            # Generate predictions
            predicted_prices = self.model.predict(forecast_df)
            
            # Calculate trend
            if len(predicted_prices) > 1:
                if predicted_prices[-1] > predicted_prices[0] * 1.05:
                    trend = "increasing"
                elif predicted_prices[-1] < predicted_prices[0] * 0.95:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "unknown"
            
            # Format results
            forecast_results = {
                "crop": crop,
                "region": region,
                "forecast_period": forecast_period,
                "start_date": today.strftime("%Y-%m-%d"),
                "end_date": forecast_dates[-1].strftime("%Y-%m-%d"),
                "price_trend": trend,
                "daily_forecasts": [
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "forecasted_price": float(price)
                    }
                    for date, price in zip(forecast_dates, predicted_prices)
                ],
                "average_price": float(np.mean(predicted_prices)),
                "min_price": float(np.min(predicted_prices)),
                "max_price": float(np.max(predicted_prices))
            }
            
            # Save forecast to database
            self._save_forecast(forecast_results)
            
            return {
                "status": "success",
                "forecast": forecast_results
            }
            
        except Exception as e:
            self.logger.error(f"Error forecasting prices: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _estimate_prices_without_model(self, crop, region, forecast_period):
        """Fallback method when insufficient data is available"""
        self.logger.info(f"Using fallback estimation for {crop} in {region}")
        
        # Get average price from market_insights or use default
        try:
            conn = self._connect_to_db()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT AVG(current_price) FROM market_insights WHERE crop = ?",
                (crop,)
            )
            result = cursor.fetchone()
            base_price = result[0] if result[0] else 2.50  # Default price if no data
            conn.close()
        except:
            base_price = 2.50  # Default fallback price
        
        # Simple estimation with slight random variation
        today = datetime.now()
        forecast_dates = [today + timedelta(days=i) for i in range(forecast_period)]
        
        np.random.seed(hash(f"{crop}_{region}") % 10000)  # Deterministic randomness
        variations = np.random.uniform(0.95, 1.05, size=forecast_period)
        predicted_prices = base_price * variations
        
        # Format results
        forecast_results = {
            "crop": crop,
            "region": region,
            "forecast_period": forecast_period,
            "start_date": today.strftime("%Y-%m-%d"),
            "end_date": forecast_dates[-1].strftime("%Y-%m-%d"),
            "price_trend": "stable",
            "daily_forecasts": [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "forecasted_price": float(price)
                }
                for date, price in zip(forecast_dates, predicted_prices)
            ],
            "average_price": float(np.mean(predicted_prices)),
            "min_price": float(np.min(predicted_prices)),
            "max_price": float(np.max(predicted_prices)),
            "note": "Limited historical data available. Using estimation."
        }
        
        return {
            "status": "success", 
            "forecast": forecast_results,
            "warning": "Limited data available. Results are estimates only."
        }
    
    def _save_forecast(self, forecast):
        """Save forecast results to the database"""
        try:
            conn = self._connect_to_db()
            cursor = conn.cursor()
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crop TEXT,
                region TEXT,
                start_date TEXT,
                end_date TEXT,
                average_price REAL,
                price_trend TEXT,
                forecast_data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            import json
            forecast_json = json.dumps(forecast)
            
            cursor.execute(
                """INSERT INTO price_forecasts 
                   (crop, region, start_date, end_date, average_price, price_trend, forecast_data) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    forecast["crop"],
                    forecast["region"],
                    forecast["start_date"],
                    forecast["end_date"],
                    forecast["average_price"],
                    forecast["price_trend"],
                    forecast_json
                )
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving forecast: {str(e)}")
    
    def analyze_demand_trends(self, crop, region):
        """
        Analyze demand trends for specific crops in the given region.
        
        Args:
            crop (str): Crop name
            region (str): Target region
        
        Returns:
            dict: Demand trend analysis
        """
        self.logger.info(f"Analyzing demand trends for {crop} in {region}")
        
        try:
            # Get historical data
            conn = self._connect_to_db()
            historic_demand = pd.read_sql(
                f"SELECT * FROM market_insights WHERE crop = '{crop}' AND region = '{region}' ORDER BY timestamp",
                conn
            )
            conn.close()
            
            if len(historic_demand) < 3:
                return {
                    "status": "warning",
                    "message": f"Insufficient historical data for {crop} in {region}",
                    "trend": "unknown"
                }
            
            # Simple trend analysis
            recent_demand = historic_demand.iloc[-3:]['demand_trend'].value_counts().idxmax()
            
            # More sophisticated analysis would include seasonality, external factors, etc.
            
            result = {
                "crop": crop,
                "region": region,
                "current_demand": recent_demand,
                "demand_forecast": self._forecast_future_demand(historic_demand),
                "factors": self._identify_demand_factors(crop, region)
            }
            
            return {"status": "success", "analysis": result}
            
        except Exception as e:
            self.logger.error(f"Error analyzing demand trends: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _forecast_future_demand(self, historic_demand):
        """Simple demand forecasting based on historical patterns"""
        if len(historic_demand) < 5:
            return "stable"
            
        # Extract demand trend values and convert to numeric
        demand_mapping = {"high": 3, "medium": 2, "low": 1}
        try:
            numeric_demand = historic_demand['demand_trend'].map(demand_mapping)
            trend = np.polyfit(range(len(numeric_demand)), numeric_demand, 1)[0]
            
            if trend > 0.1:
                return "increasing"
            elif trend < -0.1:
                return "decreasing"
            else:
                return "stable"
        except:
            return "stable"
    
    def _identify_demand_factors(self, crop, region):
        """Identify key factors influencing demand"""
        # This would ideally use more complex analysis
        # For now, return generic factors based on crop type
        
        common_factors = ["seasonal patterns", "market competition"]
        
        crop_specific = {
            "corn": ["ethanol production", "livestock feed demand"],
            "wheat": ["global wheat reserves", "bread consumption trends"],
            "soybean": ["vegetable oil market", "protein meal demand"],
            "rice": ["asian import/export policies", "dietary shifts"],
            "cotton": ["textile industry trends", "synthetic fiber prices"],
        }
        
        factors = common_factors
        for crop_key in crop_specific:
            if crop_key in crop.lower():
                factors.extend(crop_specific[crop_key])
                break
        
        return factors[:3]  # Return top 3 factors
    
    def get_sustainability_impact(self, crop, region):
        """
        Assess the environmental sustainability impact of growing specific crops.
        
        Args:
            crop (str): Crop name
            region (str): Target region
        
        Returns:
            dict: Sustainability metrics
        """
        # This would ideally be based on real data and models
        # For now, use simplified estimates
        
        crop_footprints = {
            "corn": {"water": "high", "carbon": "medium", "soil": "medium"},
            "wheat": {"water": "medium", "carbon": "low", "soil": "medium"},
            "soybean": {"water": "medium", "carbon": "low", "soil": "good"},
            "rice": {"water": "very high", "carbon": "high", "soil": "medium"},
            "cotton": {"water": "high", "carbon": "medium", "soil": "poor"},
            "potato": {"water": "medium", "carbon": "low", "soil": "poor"},
            "tomato": {"water": "high", "carbon": "medium", "soil": "medium"},
        }
        
        region_modifiers = {
            "california": {"water": +1, "carbon": 0, "soil": 0},
            "midwest": {"water": -1, "carbon": 0, "soil": +1},
            "southeast": {"water": 0, "carbon": +1, "soil": -1},
            "northwest": {"water": -1, "carbon": -1, "soil": 0},
        }
        
        # Default footprint if crop not found
        footprint = crop_footprints.get(crop.lower(), {"water": "medium", "carbon": "medium", "soil": "medium"})
        
        # Apply regional modifiers
        region_mod = {}
        for r in region_modifiers:
            if r in region.lower():
                region_mod = region_modifiers[r]
                break
        
        # Convert text ratings to numeric
        rating_map = {"very high": 5, "high": 4, "medium": 3, "low": 2, "very low": 1, 
                     "poor": 1, "fair": 2, "medium": 3, "good": 4, "excellent": 5}
        
        # Calculate adjusted ratings
        water_score = max(1, min(5, rating_map[footprint["water"]] + region_mod.get("water", 0)))
        carbon_score = max(1, min(5, rating_map[footprint["carbon"]] + region_mod.get("carbon", 0)))
        soil_score = max(1, min(5, rating_map[footprint["soil"]] + region_mod.get("soil", 0)))
        
        # Convert back to text
        reverse_map = {1: "very low", 2: "low", 3: "medium", 4: "high", 5: "very high"}
        soil_map = {1: "poor", 2: "fair", 3: "medium", 4: "good", 5: "excellent"}
        
        return {
            "crop": crop,
            "region": region,
            "water_usage": reverse_map[water_score],
            "carbon_footprint": reverse_map[carbon_score],
            "soil_impact": soil_map[soil_score],
            "overall_sustainability": round((6 - water_score + 6 - carbon_score + soil_score) / 3, 1),
            "recommendations": self._get_sustainability_recommendations(crop, region, water_score, carbon_score, soil_score)
        }
    
    def _get_sustainability_recommendations(self, crop, region, water_score, carbon_score, soil_score):
        """Generate sustainability recommendations based on scores"""
        recommendations = []
        
        if water_score >= 4:
            recommendations.append("Consider implementing drip irrigation to reduce water usage")
            
        if carbon_score >= 4:
            recommendations.append("Look into cover cropping or no-till practices to reduce carbon footprint")
            
        if soil_score <= 2:
            recommendations.append("Implement crop rotation or consider organic amendments to improve soil health")
            
        if not recommendations:
            recommendations.append("Current crop choice is relatively sustainable for this region")
            
        return recommendations