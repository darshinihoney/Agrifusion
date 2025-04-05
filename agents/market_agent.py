import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta

class MarketDynamicsAgent:
    """
    Agent that analyzes regional market trends, crop pricing, and demand forecasts
    to suggest the most profitable crops to plant.
    """
    
    def __init__(self, market_forecaster, historical_data):
        """
        Initialize the Market Dynamics Agent
        
        Args:
            market_forecaster: The market forecasting model
            historical_data: Historical market data for reference
        """
        self.market_forecaster = market_forecaster
        self.historical_data = historical_data
        self.db_connection = sqlite3.connect('agrifusion.db')
    
    def __del__(self):
        """Close database connection when object is destroyed"""
        if hasattr(self, 'db_connection'):
            self.db_connection.close()
    
    def store_market_insight(self, insight):
        """
        Store market insight in the database
        
        Args:
            insight: Dictionary containing market insight information
        """
        cursor = self.db_connection.cursor()
        
        cursor.execute("""
        INSERT INTO market_insights (crop, current_price, forecasted_price, demand_trend, region)
        VALUES (?, ?, ?, ?, ?)
        """, (
            insight['crop'],
            insight['current_price'],
            insight['forecasted_price'],
            insight['demand_trend'],
            insight['region']
        ))
        
        self.db_connection.commit()
    
    def get_previous_insights(self, region, limit=5):
        """
        Retrieve recent market insights from the database
        
        Args:
            region: Region to get insights for
            limit: Maximum number of insights to retrieve
            
        Returns:
            List of market insights
        """
        cursor = self.db_connection.cursor()
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
            insights.append({
                'crop': row[0],
                'current_price': row[1],
                'forecasted_price': row[2],
                'demand_trend': row[3],
                'timestamp': row[4]
            })
        
        return insights
    
    def analyze_market(self, region):
        """
        Analyze market conditions for a specific region
        
        Args:
            region: The region to analyze
            
        Returns:
            Dictionary containing market analysis results
        """
        # Filter historical data for the specified region
        regional_data = self.historical_data[self.historical_data['region'] == region]
        
        if len(regional_data) == 0:
            return {"error": f"No market data available for region: {region}"}
        
        # Get market forecasts for the region
        market_forecast = self.market_forecaster.forecast_prices(region)
        
        # Get highest potential profit crops
        profit_potential = self._calculate_profit_potential(regional_data, market_forecast)
        
        # Analyze market stability
        market_stability = self._analyze_market_stability(regional_data)
        
        # Evaluate demand trends
        demand_trends = self._evaluate_demand_trends(regional_data, market_forecast)
        
        # Find top 3 crops by profit potential
        top_crops = sorted(profit_potential.items(), key=lambda x: x[1]['potential'], reverse=True)[:3]
        
        # Generate recommendations
        recommendations = []
        for crop, data in top_crops:
            # Get stability info
            stability = market_stability.get(crop, {'stability': 'unknown', 'volatility': 0})
            
            # Get demand trend
            demand = demand_trends.get(crop, 'stable')
            
            # Create recommendation
            recommendation = {
                'crop': crop,
                'current_price': data['current_price'],
                'forecasted_price': data['forecasted_price'],
                'profit_potential': data['potential'],
                'market_stability': stability['stability'],
                'price_volatility': stability['volatility'],
                'demand_trend': demand,
                'region': region
            }
            
            recommendations.append(recommendation)
            
            # Store in database
            self.store_market_insight({
                'crop': crop,
                'current_price': data['current_price'],
                'forecasted_price': data['forecasted_price'],
                'demand_trend': demand,
                'region': region
            })
        
        # Get previous insights for comparison
        previous_insights = self.get_previous_insights(region)
        
        # Return comprehensive analysis
        return {
            'region': region,
            'top_recommendations': recommendations,
            'market_overview': self._generate_market_overview(regional_data, market_forecast),
            'previous_insights': previous_insights
        }
    
    def _calculate_profit_potential(self, regional_data, market_forecast):
        """
        Calculate profit potential for different crops based on current and forecasted prices
        
        Args:
            regional_data: DataFrame containing regional market data
            market_forecast: Dictionary containing forecasted prices
            
        Returns:
            Dictionary mapping crops to their profit potential
        """
        profit_potential = {}
        
        # Get unique crops in the region
        crops = regional_data['crop'].unique()
        
        for crop in crops:
            # Get current price (most recent data)
            crop_data = regional_data[regional_data['crop'] == crop].sort_values('date', ascending=False)
            
            if len(crop_data) > 0:
                current_price = crop_data.iloc[0]['price']
                
                # Get forecasted price
                if crop in market_forecast:
                    forecasted_price = market_forecast[crop]
                    
                    # Calculate potential percentage increase
                    if current_price > 0:
                        potential = ((forecasted_price - current_price) / current_price) * 100
                    else:
                        potential = 0
                        
                    profit_potential[crop] = {
                        'current_price': current_price,
                        'forecasted_price': forecasted_price,
                        'potential': potential
                    }
        
        return profit_potential
    
    def _analyze_market_stability(self, regional_data):
        """
        Analyze market stability for different crops
        
        Args:
            regional_data: DataFrame containing regional market data
            
        Returns:
            Dictionary mapping crops to their market stability metrics
        """
        stability = {}
        
        # Get unique crops in the region
        crops = regional_data['crop'].unique()
        
        for crop in crops:
            # Get crop data sorted by date
            crop_data = regional_data[regional_data['crop'] == crop].sort_values('date')
            
            if len(crop_data) > 5:  # Need sufficient data points
                # Calculate price volatility (standard deviation of percentage changes)
                prices = crop_data['price'].values
                price_changes = [(prices[i] - prices[i-1]) / prices[i-1] * 100 for i in range(1, len(prices))]
                volatility = pd.Series(price_changes).std()
                
                # Determine stability category
                if volatility < 5:
                    stability_category = 'very stable'
                elif volatility < 10:
                    stability_category = 'stable'
                elif volatility < 20:
                    stability_category = 'moderate'
                else:
                    stability_category = 'volatile'
                
                stability[crop] = {
                    'stability': stability_category,
                    'volatility': round(volatility, 2)
                }
            else:
                stability[crop] = {
                    'stability': 'unknown',
                    'volatility': 0
                }
        
        return stability
    
    def _evaluate_demand_trends(self, regional_data, market_forecast):
        """
        Evaluate demand trends for different crops
        
        Args:
            regional_data: DataFrame containing regional market data
            market_forecast: Dictionary containing forecasted prices
            
        Returns:
            Dictionary mapping crops to their demand trend
        """
        demand_trends = {}
        
        # Get unique crops in the region
        crops = regional_data['crop'].unique()
        
        for crop in crops:
            # Get crop data sorted by date
            crop_data = regional_data[regional_data['crop'] == crop].sort_values('date')
            
            if len(crop_data) >= 12:  # Need at least a year of data
                # Calculate 3-month moving average of volume
                crop_data['volume_ma'] = crop_data['volume'].rolling(window=3).mean()
                
                # Get most recent complete data
                recent_data = crop_data.dropna().tail(6)
                
                if len(recent_data) >= 6:
                    # Calculate volume trend
                    volume_start = recent_data['volume_ma'].iloc[0]
                    volume_end = recent_data['volume_ma'].iloc[-1]
                    
                    if volume_end > volume_start * 1.1:
                        demand_trend = 'increasing'
                    elif volume_end < volume_start * 0.9:
                        demand_trend = 'decreasing'
                    else:
                        demand_trend = 'stable'
                    
                    demand_trends[crop] = demand_trend
                else:
                    demand_trends[crop] = 'stable'
            else:
                demand_trends[crop] = 'stable'
        
        return demand_trends
    
    def _generate_market_overview(self, regional_data, market_forecast):
        """
        Generate a general market overview for the region
        
        Args:
            regional_data: DataFrame containing regional market data
            market_forecast: Dictionary containing forecasted prices
            
        Returns:
            Dictionary containing market overview
        """
        # Get most recent data
        recent_data = regional_data.sort_values('date', ascending=False)
        
        # Calculate average price change
        avg_price_change = 0
        count = 0
        
        for crop in market_forecast:
            crop_data = recent_data[recent_data['crop'] == crop]
            
            if len(crop_data) > 0:
                current_price = crop_data.iloc[0]['price']
                forecasted_price = market_forecast[crop]
                
                if current_price > 0:
                    price_change = ((forecasted_price - current_price) / current_price) * 100
                    avg_price_change += price_change
                    count += 1
        
        if count > 0:
            avg_price_change /= count
        
        # Determine market trend
        if avg_price_change > 5:
            market_trend = 'bullish'
        elif avg_price_change < -5:
            market_trend = 'bearish'
        else:
            market_trend = 'stable'
        
        # Get top volume crops
        total_volumes = regional_data.groupby('crop')['volume'].sum().sort_values(ascending=False)
        top_volume_crops = total_volumes.index[:3].tolist()
        
        return {
            'market_trend': market_trend,
            'average_price_change': round(avg_price_change, 2),
            'top_volume_crops': top_volume_crops
        }