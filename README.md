# 🌾 AgriFusion: Intelligent Agriculture Platform

**AgriFusion** transforms modern agriculture through AI-powered decision-making that connects farmers with market intelligence.

## 💡 Key Features

- **Smart Crop Planning** - Data-driven recommendations for optimal crop selection
- **Market Intelligence** - Real-time price forecasting and market strategy optimization
- **Sustainability Metrics** - Environmental impact analysis and resource conservation suggestions
- **Multi-Agent Simulation** - Realistic modeling of agricultural ecosystem dynamics

## 🔧 Tech Stack

| Category              | Tools Used                                   |
|-----------------------|----------------------------------------------|
| Multi-Agent Framework | Mesa     
| ML & Forecasting      | Scikit-learn, Prophet                        |
| Data Handling         | Pandas, NumPy                                |
| API Framework         | FastAPI                                      |
| DB Management         | SQLite                                       |
| Visualization         | Matplotlib, Seaborn                          |

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/agrifusion.git
cd agrifusion

# Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## 📊 Data Sources

AgriFusion integrates two specialized datasets:
| Dataset | Contents | Purpose |
|---------|----------|---------|
| 🌱 Farm Data | Soil composition, weather patterns, crop yield | Powers the Farmer Intelligence Agent |
| 📈 Market Data | Price history, supply-demand metrics, economic indicators | Drives the Market Dynamics Agent |

## 🧠 Agent System

```
┌─────────────────────┐      ┌──────────────────────┐
│                     │      │                      │
│  Farmer Agent       │◄────►│  Market Agent        │
│  - Crop optimization│      │  - Price forecasting │
│  - Resource planning│      │  - Trend analysis    │
│                     │      │                      │
└────────┬────────────┘      └──────────┬───────────┘
         │                              │
         │                              │
         ▼                              ▼
┌─────────────────────────────────────────────────────┐
│                                                     │
│               Decision Engine                       │
│         Optimized farming recommendations           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 📱 Usage Examples

```python
# Get personalized crop recommendations
recommendations = farmer_agent.get_recommendations(farm_id=1234)

# View market forecasts
forecast = market_agent.predict_prices(crop="wheat", timeframe=30)

# Run full ecosystem simulation
simulation = AgriFusionSimulation(steps=180)
results = simulation.run()
```

## 🔄 Workflow

1. **Data Integration** → **Agent Modeling** → **Forecasting** → **Decision Engine** → **API Access**


## 🤝 Contributing

We welcome contributions! See our [contributing guidelines](CONTRIBUTING.md) for details on submitting pull requests.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
