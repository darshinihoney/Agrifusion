import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sqlite3
import logging
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates

class InsightsVisualizer:
    """
    The InsightsVisualizer creates visual representations of agricultural
    and market data to help farmers make informed decisions.
    """
    
    def __init__(self, output_dir='./visualizations/output'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str): Directory to save visualization outputs
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def _connect_to_db(self):
        """Create a connection to the SQLite database"""
        return sqlite3.connect('agrifusion.db')
    
    def plot_market_trends(self, market_data, save=True):
        """
        Plot price trends and market insights for agricultural commodities.
        
        Args:
            market_data (dict): Market data including price forecasts
            save (bool): Whether to save the visualization to disk
            
        Returns:
            str: Path to saved visualization if save=True, otherwise None
        """
        self.logger.info("Generating market trends visualization")
        
        try:
            # Extract data from the market_data dict
            crop = market_data.get('crop', 'Unknown Crop')
            region = market_data.get('region', 'Unknown Region')
            forecast = market_data.get('forecast', {})
            daily_forecasts = forecast.get('daily_forecasts', [])
            
            if not daily_forecasts:
                self.logger.warning("No forecast data available for visualization")
                return None
            
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(daily_forecasts)
            df['date'] = pd.to_datetime(df['date'])
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Price forecast line chart
            ax1.plot(df['date'], df['forecasted_price'], marker='o', linestyle='-', color='#3366cc', 
                    linewidth=2, markersize=6)
            
            # Add trend line
            z = np.polyfit(range(len(df)), df['forecasted_price'], 1)
            p = np.poly1d(z)
            ax1.plot(df['date'], p(range(len(df))), "r--", alpha=0.8)
            
            # Format date axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Add price range
            min_price = min(df['forecasted_price'])
            max_price = max(df['forecasted_price'])
            ax1.axhspan(min_price, max_price, alpha=0.2, color='#3366cc')
            
            # Add labels and title
            ax1.set_title(f'Price Forecast for {crop} in {region}', fontsize=16)
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Annotate min and max points
            min_idx = df['forecasted_price'].idxmin()
            max_idx = df['forecasted_price'].idxmax()
            
            ax1.annotate(f'Min: ${df["forecasted_price"][min_idx]:.2f}', 
                        xy=(df['date'][min_idx], df['forecasted_price'][min_idx]),
                        xytext=(10, -30), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
                        
            ax1.annotate(f'Max: ${df["forecasted_price"][max_idx]:.2f}', 
                        xy=(df['date'][max_idx], df['forecasted_price'][max_idx]),
                        xytext=(10, 30), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            
            # Plot 2: Get historical data from database for comparison
            try:
                conn = self._connect_to_db()
                historic_data = pd.read_sql_query(
                    f"SELECT * FROM market_insights WHERE crop = '{crop}' AND region = '{region}' ORDER BY timestamp DESC LIMIT 90",
                    conn
                )
                conn.close()
                
                if len(historic_data) > 0:
                    # Convert timestamp to datetime
                    historic_data['timestamp'] = pd.to_datetime(historic_data['timestamp'])
                    
                    # Plot historical prices
                    ax2.plot(historic_data['timestamp'], historic_data['current_price'], 
                            marker='s', linestyle='-', color='#33aa33', 
                            linewidth=2, markersize=6, label='Historical Prices')
                    
                    # Add forecasted prices
                    ax2.plot(df['date'], df['forecasted_price'], 
                            marker='o', linestyle='--', color='#3366cc', 
                            linewidth=2, markersize=6, label='Forecasted Prices')
                    
                    # Add vertical line to separate historical and forecasted data
                    current_date = datetime.now()
                    ax2.axvline(x=current_date, color='r', linestyle='-', alpha=0.5)
                    ax2.annotate('Today', xy=(current_date, ax2.get_ylim()[0]),
                                xytext=(0, -30), textcoords='offset points',
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                                ha='center')
                    
                    # Format date axis
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=15))
                    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
                    
                    # Add labels and title
                    ax2.set_title('Historical vs Forecasted Prices', fontsize=16)
                    ax2.set_xlabel('Date', fontsize=12)
                    ax2.set_ylabel('Price ($)', fontsize=12)
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(loc='best')
                    
                else:
                    # If no historical data, show information
                    ax2.text(0.5, 0.5, "No historical data available for comparison",
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax2.transAxes, fontsize=14)
                    ax2.set_title('Historical Data', fontsize=16)
            
            except Exception as e:
                self.logger.error(f"Error plotting historical data: {str(e)}")
                ax2.text(0.5, 0.5, f"Error retrieving historical data: {str(e)}",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=12)
            
            # Adjust layout and spacing
            plt.tight_layout()
            
            # Save the figure if requested
            if save:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{self.output_dir}/{crop}_{region}_market_trends_{timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                self.logger.info(f"Market trends visualization saved to {filename}")
                return filename
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating market trends visualization: {str(e)}")
            return None
    
    def plot_crop_comparison(self, crop_data, metric='profitability', top_n=5, save=True):
        """
        Create a comparison chart of crops based on selected metrics.
        
        Args:
            crop_data (list): List of crop data dictionaries
            metric (str): Metric to compare (profitability, water_usage, sustainability)
            top_n (int): Number of top crops to display
            save (bool): Whether to save the visualization
            
        Returns:
            str: Path to saved visualization if save=True, otherwise None
        """
        self.logger.info(f"Generating crop comparison visualization for {metric}")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(crop_data)
            
            if len(df) == 0:
                self.logger.warning("No crop data available for comparison")
                return None
                
            # Sort by the selected metric
            if metric in df.columns:
                df = df.sort_values(by=metric, ascending=False).head(top_n)
            else:
                self.logger.warning(f"Metric '{metric}' not found in crop data")
                return None
            
            # Create the bar chart
            plt.figure(figsize=(12, 8))
            bars = plt.bar(df['crop_name'], df[metric], color=sns.color_palette("viridis", len(df)))
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom')
            
            # Customize the chart
            title_mapping = {
                'profitability': 'Crop Profitability Comparison',
                'water_usage': 'Crop Water Usage Comparison',
                'sustainability': 'Crop Sustainability Score Comparison'
            }
            y_label_mapping = {
                'profitability': 'Estimated Profit ($/acre)',
                'water_usage': 'Water Usage (gallons/acre)',
                'sustainability': 'Sustainability Score (higher is better)'
            }
            
            plt.title(title_mapping.get(metric, f'Crop {metric.capitalize()} Comparison'), fontsize=16)
            plt.xlabel('Crop', fontsize=14)
            plt.ylabel(y_label_mapping.get(metric, metric.capitalize()), fontsize=14)
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            
            # Add explanatory notes based on metric
            if metric == 'sustainability':
                plt.figtext(0.5, 0.01, 
                        "Sustainability score combines water efficiency, carbon footprint, and soil health",
                        ha='center', fontsize=10, style='italic')
            elif metric == 'profitability':
                plt.figtext(0.5, 0.01, 
                        "Profitability estimates based on current market prices and typical yields",
                        ha='center', fontsize=10, style='italic')
            
            plt.tight_layout()
            
            # Save the figure if requested
            if save:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{self.output_dir}/crop_comparison_{metric}_{timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                self.logger.info(f"Crop comparison visualization saved to {filename}")
                return filename
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating crop comparison visualization: {str(e)}")
            return None
    
    def plot_sustainability_radar(self, farm_data, save=True):
        """
        Create a radar chart showing sustainability metrics for a farm.
        
        Args:
            farm_data (dict): Farm sustainability data
            save (bool): Whether to save the visualization
            
        Returns:
            str: Path to saved visualization if save=True, otherwise None
        """
        self.logger.info("Generating sustainability radar chart")
        
        try:
            # Extract sustainability metrics
            categories = ['Water Efficiency', 'Carbon Footprint', 'Soil Health', 
                          'Biodiversity', 'Chemical Usage', 'Energy Efficiency']
            
            # Convert to numeric values (0-100 scale, higher is better)
            values = [
                farm_data.get('water_efficiency', 50),
                100 - farm_data.get('carbon_footprint', 50),  # Invert carbon footprint (lower is better)
                farm_data.get('soil_health', 50),
                farm_data.get('biodiversity', 50),
                100 - farm_data.get('chemical_usage', 50),  # Invert chemical usage (lower is better)
                farm_data.get('energy_efficiency', 50)
            ]
            
            # Number of variables
            N = len(categories)
            
            # Create angles for each category
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add values to complete the loop
            values += values[:1]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            # Draw the chart
            ax.plot(angles, values, linewidth=2, linestyle='solid', color='#3366cc')
            ax.fill(angles, values, color='#3366cc', alpha=0.25)
            
            # Add category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=12)
            
            # Add value labels at each point
            for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
                ax.text(angle, value + 10, f"{value}", 
                        horizontalalignment='center', verticalalignment='center')
            
            # Set y-ticks and limits
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_ylim(0, 100)
            
            # Add chart title and legend
            plt.title(f"Sustainability Profile: {farm_data.get('farm_name', 'Farm')}", size=16, y=1.1)
            
            # Add explanatory note
            plt.figtext(0.5, 0.01, 
                      "Higher values indicate better sustainability performance. Carbon footprint and chemical usage are inverted.",
                      ha='center', fontsize=10, style='italic')
            
            # Save the figure if requested
            if save:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                farm_name = farm_data.get('farm_name', 'farm').replace(' ', '_').lower()
                filename = f"{self.output_dir}/{farm_name}_sustainability_{timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                self.logger.info(f"Sustainability radar chart saved to {filename}")
                return filename
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating sustainability radar chart: {str(e)}")
            return None
        
    def plot_crop_calendar(self, crop_recommendations, farm_location, save=True):
        """
        Create a visualization of the recommended planting and harvesting schedule.
        
        Args:
            crop_recommendations (list): List of recommended crops with timing data
            farm_location (str): Farm location for title
            save (bool): Whether to save the visualization
            
        Returns:
            str: Path to saved visualization if save=True, otherwise None
        """
        self.logger.info("Generating crop calendar visualization")
        
        try:
            # Convert recommendations to DataFrame if needed
            if not isinstance(crop_recommendations, pd.DataFrame):
                df = pd.DataFrame(crop_recommendations)
            else:
                df = crop_recommendations.copy()
            
            if len(df) == 0:
                self.logger.warning("No crop recommendations available for calendar")
                return None
            
            # Ensure required columns exist
            required_cols = ['crop_name', 'planting_start', 'planting_end', 'harvest_start', 'harvest_end']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Missing required columns for crop calendar: {missing_cols}")
                return None
            
            # Convert date strings to datetime objects if needed
            date_cols = ['planting_start', 'planting_end', 'harvest_start', 'harvest_end']
            for col in date_cols:
                if df[col].dtype == object:  # If dates are strings
                    df[col] = pd.to_datetime(df[col])
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Define colors for planting and harvesting
            planting_color = '#66c2a5'  # Green for planting
            harvest_color = '#fc8d62'   # Orange for harvesting
            
            # Get current year
            current_year = datetime.now().year
            
            # Create month labels for x-axis
            months = pd.date_range(start=f'{current_year}-01-01', periods=12, freq='MS')
            month_labels = [m.strftime('%b') for m in months]
            
            # Set up the plot
            ax.set_xlim(0, 12)
            ax.set_xticks(range(12))
            ax.set_xticklabels(month_labels)
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df['crop_name'])
            ax.grid(True, axis='x', alpha=0.3)
            
            # Plot horizontal bars for each crop
            for i, row in df.iterrows():
                # Calculate month positions (0-11) for date ranges
                planting_start_month = row['planting_start'].month - 1
                planting_end_month = row['planting_end'].month - 1
                harvest_start_month = row['harvest_start'].month - 1
                harvest_end_month = row['harvest_end'].month - 1
                
                # Handle year transitions (e.g., planting from Nov to Feb)
                if planting_end_month < planting_start_month:
                    planting_end_month += 12
                if harvest_end_month < harvest_start_month:
                    harvest_end_month += 12
                
                # Calculate durations
                planting_duration = planting_end_month - planting_start_month + 1
                harvest_duration = harvest_end_month - harvest_start_month + 1
                
                # Plot planting period
                ax.barh(i, planting_duration, left=planting_start_month, height=0.5, 
                       color=planting_color, alpha=0.7, label='Planting' if i == 0 else "")
                
                # Plot harvest period
                ax.barh(i, harvest_duration, left=harvest_start_month, height=0.5, 
                       color=harvest_color, alpha=0.7, label='Harvest' if i == 0 else "")
                
                # Add crop yield or other details if available
                if 'expected_yield' in df.columns:
                    yield_text = f" (Est. yield: {row['expected_yield']} units/acre)"
                    ax.text(11.5, i, yield_text, va='center', ha='right', fontsize=8)
            
            # Add a vertical line for current month
            current_month = datetime.now().month - 1  # 0-based index
            ax.axvline(x=current_month, color='red', linestyle='--', alpha=0.7)
            ax.text(current_month, -0.5, 'Current month', rotation=90, 
                   color='red', va='top', ha='right')
            
            # Add legend and only show once for each category
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            # Add title and labels
            ax.set_title(f'Crop Calendar for {farm_location}', fontsize=16)
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Recommended Crops', fontsize=12)
            
            # Add explanatory note
            fig.text(0.5, 0.01, 
                    "Green bars indicate planting periods, orange bars indicate harvest periods",
                    ha='center', fontsize=10, style='italic')
            
            # Adjust layout
            fig.tight_layout()
            
            # Save the figure if requested
            if save:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                location = farm_location.replace(' ', '_').lower()
                filename = f"{self.output_dir}/{location}_crop_calendar_{timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                self.logger.info(f"Crop calendar visualization saved to {filename}")
                return filename
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating crop calendar visualization: {str(e)}")
            return None