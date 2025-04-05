import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import os

class WeatherPredictor:
    """
    Model for predicting weather conditions relevant to agricultural planning
    """
    
    def __init__(self):
        """Initialize the weather predictor"""
        # Weather API credentials (mock - would be loaded from environment variables in production)
        self.api_key = os.environ.get('WEATHER_API_KEY', 'mock_key')
        
        # Cache for weather data to reduce API calls
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=6)  # Cache expires after 6 hours
    
    def get_forecast(self, location, days=30):
        """
        Get weather forecast for a specific location
        
        Args:
            location: String location name or coordinates
            days: Number of days to forecast
            
        Returns:
            Dictionary containing weather forecast data
        """
        current_time = datetime.now()
        
        # Check if we have valid cached data
        cache_key = f"{location}_{days}"
        if cache_key in self.cache and current_time < self.cache_expiry.get(cache_key, current_time):
            return self.cache[cache_key]
        
        # In a real implementation, this would call a weather API
        # For this demo, we'll generate mock data
        forecast = self._generate_mock_forecast(location, days)
        
        # Cache the result
        self.cache[cache_key] = forecast
        self.cache_expiry[cache_key] = current_time + self.cache_duration
        
        return forecast
    
    def _generate_mock_forecast(self, location, days):
        """
        Generate mock weather forecast data
        
        Args:
            location: String location name or coordinates
            days: Number of days to forecast
            
        Returns:
            Dictionary containing mock weather forecast data
        """
        # Parse location string to extract latitude and longitude if provided
        lat, lon = self._parse_location(location)
        
        # Generate forecast based on location and time of year
        current_date = datetime.now()
        month = current_date.month
        
        # Base temperature varies by latitude (hotter near equator)
        base_temp = max(5, 30 - abs(lat) * 0.5)
        
        # Seasonal adjustment (hotter in summer, colder in winter)
        # Northern hemisphere: Summer (Jun-Aug), Winter (Dec-Feb)
        # Southern hemisphere: Summer (Dec-Feb), Winter (Jun-Aug)
        if (month >= 6 and month <= 8 and lat > 0) or (month >= 12 or month <= 2 and lat < 0):
            # Summer
            temp_adjustment = 10
        elif (month >= 12 or month <= 2 and lat > 0) or (month >= 6 and month <= 8 and lat < 0):
            # Winter
            temp_adjustment = -10
        else:
            # Spring/Fall
            temp_adjustment = 0
        
        # Adjust base temperature for season
        base_temp += temp_adjustment
        
        # Generate daily temperatures with some randomness
        daily_temps = [
            round(base_temp + np.random.normal(0, 3), 1) 
            for _ in range(days)
        ]
        
        # Generate precipitation data (more rain closer to equator and oceans)
        # Simple model - could be much more sophisticated
        base_precip = 2 + abs(lon - 100) * 0.02  # More rain closer to coasts
        if abs(lat) < 15:  # Tropical regions get more rain
            base_precip *= 2
            
        daily_precip = [
            max(0, round(base_precip * np.random.exponential(1), 1))
            for _ in range(days)
        ]
        
        # Calculate average temperature and total precipitation
        avg_temperature = round(sum(daily_temps) / len(daily_temps), 1)
        total_precipitation = round(sum(daily_precip), 1)
        
        # Put everything in a dictionary
        forecast = {
            'location': location,
            'latitude': lat,
            'longitude': lon,
            'forecast_days': days,
            'avg_temperature': avg_temperature,
            'precipitation': total_precipitation,
            'daily_forecast': [
                {
                    'day': (current_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                    'temperature': daily_temps[i],
                    'precipitation': daily_precip[i]
                }
                for i in range(days)
            ],
            'growing_conditions': self._evaluate_growing_conditions(avg_temperature, total_precipitation)
        }
        
        return forecast
    
    def _parse_location(self, location):
        """
        Parse location string to extract latitude and longitude
        
        Args:
            location: String location name or coordinates
            
        Returns:
            Tuple of (latitude, longitude)
        """
        # Check if location is in "lat,lon" format
        if ',' in location and all(part.replace('-', '').replace('.', '').isdigit() for part in location.split(',')):
            parts = location.split(',')
            return float(parts[0]), float(parts[1])
        
        # For named locations, use a simple mapping
        # In a real implementation, this would use geocoding
        location_map = {
            'California': (36.7783, -119.4179),
            'Iowa': (41.8780, -93.0977),
            'Texas': (31.9686, -99.9018),
            'Florida': (27.6648, -81.5158),
            'Kansas': (39.0119, -98.4842),
            'Nebraska': (41.4925, -99.9018),
            'Minnesota': (46.7296, -94.6859),
            'Illinois': (40.6331, -89.3985),
            'Indiana': (40.2672, -86.1349),
            'Ohio': (40.4173, -82.9071)
        }
        
        # Default to middle of USA if location not found
        return location_map.get(location, (39.8283, -98.5795))
    
    def _evaluate_growing_conditions(self, avg_temperature, precipitation):
        """
        Evaluate the overall growing conditions based on temperature and precipitation
        
        Args:
            avg_temperature: Average temperature in Celsius
            precipitation: Total precipitation in mm
            
        Returns:
            Dictionary with evaluation of growing conditions
        """
        # Temperature evaluation
        if 15 <= avg_temperature <= 25:
            temp_condition = "ideal"
        elif 10 <= avg_temperature < 15 or 25 < avg_temperature <= 30:
            temp_condition = "good"
        elif 5 <= avg_temperature < 10 or 30 < avg_temperature <= 35:
            temp_condition = "fair"
        else:
            temp_condition = "poor"
        
        # Precipitation evaluation
        if 100 <= precipitation <= 300:
            precip_condition = "ideal"
        elif 50 <= precipitation < 100 or 300 < precipitation <= 500:
            precip_condition = "good"
        elif 20 <= precipitation < 50 or 500 < precipitation <= 800:
            precip_condition = "fair"
        else:
            precip_condition = "poor"
        
        # Overall condition is the worse of the two
        condition_rank = {"poor": 0, "fair": 1, "good": 2, "ideal": 3}
        if condition_rank[temp_condition] <= condition_rank[precip_condition]:
            overall_condition = temp_condition
        else:
            overall_condition = precip_condition
        
        # Suitable crops based on conditions
        suitable_crops = []
        if overall_condition == "ideal":
            suitable_crops = ["corn", "soybeans", "wheat", "vegetables", "fruits"]
        elif overall_condition == "good":
            suitable_crops = ["wheat", "barley", "oats", "corn", "soybeans"]
        elif overall_condition == "fair":
            suitable_crops = ["wheat", "barley", "sorghum", "millet"]
        else:  # poor
            suitable_crops = ["sorghum", "millet", "drought-resistant varieties"]
        
        return {
            "temperature_condition": temp_condition,
            "precipitation_condition": precip_condition,
            "overall_condition": overall_condition,
            "suitable_crops": suitable_crops
        }