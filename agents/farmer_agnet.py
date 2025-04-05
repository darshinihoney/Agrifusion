import sqlite3
import pandas as pd
from datetime import datetime

class FarmerIntelligenceAgent:
    """
    Agent that provides actionable insights by analyzing input from farmers
    about land, crop preferences, and financial goals.
    """
    
    def __init__(self, weather_predictor, historical_data):
        """
        Initialize the Farmer Intelligence Agent
        
        Args:
            weather_predictor: The weather prediction model
            historical_data: Historical farming data for reference
        """
        self.weather_predictor = weather_predictor
        self.historical_data = historical_data
        self.db_connection = sqlite3.connect('agrifusion.db')
    
    def __del__(self):
        """Close database connection when object is destroyed"""
        if hasattr(self, 'db_connection'):
            self.db_connection.close()
    
    def get_farm_data(self, farm_id):
        """
        Retrieve farm data from the database
        
        Args:
            farm_id: Unique identifier for the farm
            
        Returns:
            Dictionary containing farm data
        """
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM farm_data WHERE farm_id = ?", (farm_id,))
        result = cursor.fetchone()
        
        if result:
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, result))
        else:
            return None
    
    def store_farm_data(self, farm_data):
        """
        Store new farm data in the database
        
        Args:
            farm_data: Dictionary containing farm information
        """
        cursor = self.db_connection.cursor()
        
        cursor.execute("""
        INSERT INTO farm_data (farm_id, soil_type, location, size_hectares, water_source, previous_crops)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            farm_data['farm_id'],
            farm_data['soil_type'],
            farm_data['location'],
            farm_data['size_hectares'],
            farm_data['water_source'],
            farm_data['previous_crops']
        ))
        
        self.db_connection.commit()
    
    def store_recommendation(self, recommendation):
        """
        Store a new recommendation in the database
        
        Args:
            recommendation: Dictionary containing recommendation details
        """
        cursor = self.db_connection.cursor()
        
        cursor.execute("""
        INSERT INTO farmer_recommendations (
            farm_id, recommendation, crop_suggestion, planting_date, 
            expected_yield, water_usage_estimate, carbon_footprint_estimate
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            recommendation['farm_id'],
            recommendation['recommendation'],
            recommendation['crop_suggestion'],
            recommendation['planting_date'],
            recommendation['expected_yield'],
            recommendation['water_usage_estimate'],
            recommendation['carbon_footprint_estimate']
        ))
        
        self.db_connection.commit()
    
    def generate_recommendations(self, farm_id):
        """
        Generate sustainable farming recommendations based on farm data
        and environmental factors
        
        Args:
            farm_id: Unique identifier for the farm
            
        Returns:
            Dictionary containing recommendations
        """
        # Get farm data from database
        farm_data = self.get_farm_data(farm_id)
        
        if not farm_data:
            return {"error": f"No data found for farm ID: {farm_id}"}
        
        # Get weather forecast for farm location
        weather_forecast = self.weather_predictor.get_forecast(farm_data['location'])
        
        # Analyze soil type and previous crops
        soil_type = farm_data['soil_type']
        previous_crops = farm_data['previous_crops'].split(',')
        
        # Calculate water availability based on weather forecast and water source
        water_availability = self._calculate_water_availability(
            weather_forecast, 
            farm_data['water_source']
        )
        
        # Find suitable crops based on soil, weather, and water availability
        suitable_crops = self._find_suitable_crops(
            soil_type, 
            weather_forecast, 
            water_availability,
            previous_crops
        )
        
        # Generate crop rotation plan
        crop_rotation_plan = self._generate_crop_rotation(
            previous_crops, 
            suitable_crops
        )
        
        # Calculate expected yield and resource usage
        expected_yield = self._calculate_expected_yield(
            crop_rotation_plan['recommended_crop'], 
            soil_type, 
            weather_forecast
        )
        
        water_usage = self._estimate_water_usage(
            crop_rotation_plan['recommended_crop'], 
            farm_data['size_hectares']
        )
        
        carbon_footprint = self._estimate_carbon_footprint(
            crop_rotation_plan['recommended_crop'], 
            farm_data['size_hectares'],
            farm_data['soil_type']
        )
        
        # Format planting date recommendation
        optimal_planting_date = self._get_optimal_planting_date(
            crop_rotation_plan['recommended_crop'],
            weather_forecast
        )
        
        # Compile recommendations
        recommendation = {
            "farm_id": farm_id,
            "recommendation": crop_rotation_plan['recommendation'],
            "crop_suggestion": crop_rotation_plan['recommended_crop'],
            "planting_date": optimal_planting_date,
            "expected_yield": expected_yield,
            "water_usage_estimate": water_usage,
            "carbon_footprint_estimate": carbon_footprint
        }
        
        # Store recommendation in database
        self.store_recommendation(recommendation)
        
        return recommendation
    
    def _calculate_water_availability(self, weather_forecast, water_source):
        """
        Calculate water availability based on weather forecast and water source
        
        Args:
            weather_forecast: Dictionary containing weather forecast data
            water_source: String describing the farm's water source
            
        Returns:
            Float representing water availability score (0-1)
        """
        # Example implementation
        precipitation = weather_forecast.get('precipitation', 0)
        
        # Assign base value based on water source
        if water_source == 'river':
            base_availability = 0.8
        elif water_source == 'well':
            base_availability = 0.7
        elif water_source == 'rain_fed':
            base_availability = 0.5
        else:
            base_availability = 0.6
        
        # Adjust based on precipitation forecast
        availability = base_availability + (precipitation / 1000)
        
        # Ensure value is between 0 and 1
        return max(0, min(1, availability))
    
    def _find_suitable_crops(self, soil_type, weather_forecast, water_availability, previous_crops):
        """
        Find crops that are suitable for the given conditions
        
        Args:
            soil_type: String describing soil type
            weather_forecast: Dictionary containing weather forecast data
            water_availability: Float representing water availability (0-1)
            previous_crops: List of crops grown previously
            
        Returns:
            List of suitable crops
        """
        # Get relevant weather data
        avg_temp = weather_forecast.get('avg_temperature', 20)
        precipitation = weather_forecast.get('precipitation', 500)
        
        # Filter historical data based on similar conditions
        similar_conditions = self.historical_data[
            (self.historical_data['soil_type'] == soil_type) &
            (self.historical_data['avg_temperature'].between(avg_temp - 5, avg_temp + 5)) &
            (self.historical_data['precipitation'].between(precipitation - 100, precipitation + 100))
        ]
        
        if len(similar_conditions) > 0:
            # Get crops with good yields under similar conditions
            good_yield_threshold = similar_conditions['yield'].quantile(0.7)
            good_performing_crops = similar_conditions[
                similar_conditions['yield'] >= good_yield_threshold
            ]['crop'].unique().tolist()
            
            # Filter based on water requirements if water is scarce
            if water_availability < 0.6:
                water_efficient_crops = similar_conditions[
                    similar_conditions['water_usage'] < similar_conditions['water_usage'].median()
                ]['crop'].unique().tolist()
                suitable_crops = list(set(good_performing_crops) & set(water_efficient_crops))
            else:
                suitable_crops = good_performing_crops
        else:
            # Fallback options if no similar conditions in data
            if soil_type == 'clay':
                suitable_crops = ['wheat', 'rice', 'sorghum']
            elif soil_type == 'sandy':
                suitable_crops = ['potatoes', 'carrots', 'peanuts']
            elif soil_type == 'loamy':
                suitable_crops = ['corn', 'soybeans', 'vegetables']
            else:
                suitable_crops = ['millet', 'barley', 'oats']
        
        return suitable_crops
    
    def _generate_crop_rotation(self, previous_crops, suitable_crops):
        """
        Generate crop rotation recommendation to maintain soil health
        
        Args:
            previous_crops: List of crops grown previously
            suitable_crops: List of suitable crops for current conditions
            
        Returns:
            Dictionary with crop rotation recommendation
        """
        # Classify crops by type
        legumes = ['soybeans', 'peas', 'lentils', 'beans', 'clover']
        cereals = ['wheat', 'corn', 'barley', 'rice', 'oats', 'millet', 'sorghum']
        root_crops = ['potatoes', 'carrots', 'beets', 'turnips', 'radishes']
        leafy_crops = ['lettuce', 'spinach', 'kale', 'cabbage']
        
        # Determine recent crop types
        recent_crop_types = []
        for crop in previous_crops:
            if crop in legumes:
                recent_crop_types.append('legume')
            elif crop in cereals:
                recent_crop_types.append('cereal')
            elif crop in root_crops:
                recent_crop_types.append('root')
            elif crop in leafy_crops:
                recent_crop_types.append('leafy')
        
        # Recommend crop types for rotation
        if 'legume' not in recent_crop_types:
            target_types = ['legume']
        elif 'cereal' not in recent_crop_types:
            target_types = ['cereal']
        elif 'root' not in recent_crop_types:
            target_types = ['root']
        else:
            target_types = ['leafy', 'legume']
        
        # Find matching suitable crops for target types
        recommended_crops = []
        for crop in suitable_crops:
            if crop in legumes and 'legume' in target_types:
                recommended_crops.append(crop)
            elif crop in cereals and 'cereal' in target_types:
                recommended_crops.append(crop)
            elif crop in root_crops and 'root' in target_types:
                recommended_crops.append(crop)
            elif crop in leafy_crops and 'leafy' in target_types:
                recommended_crops.append(crop)
        
        # If no matches, use any suitable crop
        if not recommended_crops and suitable_crops:
            recommended_crops = suitable_crops
        
        if recommended_crops:
            # Pick first crop for simplicity (could be enhanced with more analysis)
            recommended_crop = recommended_crops[0]
            
            # Create recommendation text
            if recommended_crop in legumes:
                recommendation = f"Based on your recent crop history, we recommend planting {recommended_crop} to add nitrogen to the soil and break disease cycles."
            elif recommended_crop in cereals:
                recommendation = f"Consider planting {recommended_crop} as it will diversify your crop rotation and help manage soil nutrients."
            elif recommended_crop in root_crops:
                recommendation = f"We recommend {recommended_crop} which will help break up compacted soil and diversify your crop rotation."
            else:
                recommendation = f"Planting {recommended_crop} would be beneficial for your soil health based on your previous crop history."
        else:
            recommended_crop = "No specific crop recommended"
            recommendation = "We don't have enough data to make a specific crop recommendation. Consider soil testing for more tailored advice."
        
        return {
            "recommended_crop": recommended_crop,
            "recommendation": recommendation
        }
    
    def _calculate_expected_yield(self, crop, soil_type, weather_forecast):
        """
        Calculate expected yield based on crop, soil, and weather conditions
        
        Args:
            crop: String name of the crop
            soil_type: String describing soil type
            weather_forecast: Dictionary containing weather forecast data
            
        Returns:
            Float representing expected yield in tonnes per hectare
        """
        # Check if we have data for this crop
        crop_data = self.historical_data[self.historical_data['crop'] == crop]
        
        if len(crop_data) > 0:
            # Get average yield for this crop in similar conditions
            similar_conditions = crop_data[
                (crop_data['soil_type'] == soil_type)
            ]
            
            if len(similar_conditions) > 0:
                base_yield = similar_conditions['yield'].mean()
            else:
                base_yield = crop_data['yield'].mean()
        else:
            # Default yields if no data available
            default_yields = {
                'wheat': 3.0,
                'rice': 4.5,
                'corn': 5.5,
                'soybeans': 2.8,
                'potatoes': 25.0,
                'carrots': 30.0,
                'lettuce': 20.0,
                'beans': 2.0,
                'peas': 2.5
            }
            base_yield = default_yields.get(crop, 3.0)
        
        # Adjust for weather conditions
        temp_factor = 1.0
        if 'avg_temperature' in weather_forecast:
            avg_temp = weather_forecast['avg_temperature']
            # Most crops grow best between 15-25°C
            if 15 <= avg_temp <= 25:
                temp_factor = 1.1
            elif avg_temp < 10 or avg_temp > 30:
                temp_factor = 0.8
        
        # Return estimated yield
        return round(base_yield * temp_factor, 2)
    
    def _estimate_water_usage(self, crop, size_hectares):
        """
        Estimate water usage for the suggested crop
        
        Args:
            crop: String name of the crop
            size_hectares: Float size of the farm in hectares
            
        Returns:
            Float representing estimated water usage in cubic meters
        """
        # Average water requirements in cubic meters per hectare
        water_requirements = {
            'wheat': 4500,
            'rice': 10000,
            'corn': 6000,
            'soybeans': 5000,
            'potatoes': 5000,
            'carrots': 4000,
            'lettuce': 3500,
            'beans': 3500,
            'peas': 3500,
            'barley': 4000,
            'oats': 4000,
            'millet': 3000,
            'sorghum': 4000
        }
        
        # Get water requirement for this crop, default to average if not in dictionary
        water_per_hectare = water_requirements.get(crop, 5000)
        
        # Calculate total water usage
        total_water = water_per_hectare * size_hectares
        
        return round(total_water, 2)
    
    def _estimate_carbon_footprint(self, crop, size_hectares, soil_type):
        """
        Estimate carbon footprint for growing the suggested crop
        
        Args:
            crop: String name of the crop
            size_hectares: Float size of the farm in hectares
            soil_type: String describing soil type
            
        Returns:
            Float representing estimated carbon footprint in tonnes CO2 equivalent
        """
        # Average carbon emissions in tonnes CO2 equivalent per hectare
        # Values include cultivation, fertilizer, irrigation, etc.
        carbon_emissions = {
            'wheat': 3.0,
            'rice': 7.0,  # Rice has higher emissions due to methane from flooded fields
            'corn': 3.5,
            'soybeans': 2.0,  # Legumes fix nitrogen, reducing fertilizer needs
            'potatoes': 2.5,
            'carrots': 2.0,
            'lettuce': 1.5,
            'beans': 1.8,
            'peas': 1.8,
            'barley': 2.8,
            'oats': 2.5,
            'millet': 2.0,
            'sorghum': 2.5
        }
        
        # Get carbon emission for this crop, default to average if not in dictionary
        carbon_per_hectare = carbon_emissions.get(crop, 3.0)
        
        # Adjust based on soil type (organic matter in soil can sequester carbon)
        soil_factor = {
            'clay': 1.1,  # More tillage often needed
            'sandy': 1.0,  # Standard
            'loamy': 0.9,  # Better structure, may need less intervention
            'silty': 0.95,
            'peaty': 1.2  # Releases carbon when cultivated
        }.get(soil_type, 1.0)
        
        # Calculate total carbon footprint
        total_carbon = carbon_per_hectare * size_hectares * soil_factor
        
        return round(total_carbon, 2)
    
    def _get_optimal_planting_date(self, crop, weather_forecast):
        """
        Determine optimal planting date based on crop and weather forecast
        
        Args:
            crop: String name of the crop
            weather_forecast: Dictionary containing weather forecast data
            
        Returns:
            String representing optimal planting date range
        """
        # Default planting seasons by crop (in Northern Hemisphere)
        planting_seasons = {
            'wheat': {'spring': 'March-April', 'winter': 'September-October'},
            'rice': {'main': 'April-May'},
            'corn': {'main': 'April-May'},
            'soybeans': {'main': 'May-June'},
            'potatoes': {'main': 'March-April'},
            'carrots': {'main': 'April-May', 'fall': 'July-August'},
            'lettuce': {'spring': 'March-April', 'fall': 'August-September'},
            'beans': {'main': 'May-June'},
            'peas': {'spring': 'February-March', 'fall': 'September-October'},
            'barley': {'spring': 'March-April', 'winter': 'September-October'},
            'oats': {'main': 'March-April'},
            'millet': {'main': 'May-June'},
            'sorghum': {'main': 'May-June'}
        }
        
        # Get current month to determine which season we're in or approaching
        current_month = datetime.now().month
        
        # Choose appropriate planting season
        if crop in planting_seasons:
            seasons = planting_seasons[crop]
            
            if 'main' in seasons:
                return seasons['main']
            elif 'spring' in seasons and current_month in [1, 2, 3, 4, 5]:
                return seasons['spring']
            elif 'fall' in seasons and current_month in [7, 8, 9, 10]:
                return seasons['fall']
            elif 'winter' in seasons and current_month in [9, 10, 11, 12]:
                return seasons['winter']
            else:
                return list(seasons.values())[0]  # Return first available season
        else:
            # Default recommendation if crop not in database
            if current_month in [3, 4, 5]:
                return "April-May"
            elif current_month in [8, 9, 10]:
                return "September-October"
            else:
                return "Consult local extension office for specific planting dates"