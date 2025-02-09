# RetroTech Auto Analytics Dashboard 🚗

A retrofuturistic vehicle analytics platform that combines modern machine learning with vintage aesthetics. This project showcases real-time vehicle diagnostics with a retro-synthwave UI.

## 🌟 Features

- Real-time vehicle diagnostics monitoring
- Machine learning-powered maintenance predictions
- Retro-synthwave UI design with modern functionality
- Interactive performance metrics
- Predictive analytics for vehicle health

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Plotly, Dash
- **Styling**: Custom CSS with retro-synthwave aesthetics

## 🌐 Live Demo
Access the live demo at: [RetroTech Auto Analytics](https://retrotech-auto-analytics.onrender.com)
*(Note: Demo credentials - Username: demo, Password: demo123)*

## 🎥 Quick Demo Video
![Dashboard Preview](docs/dashboard-preview.gif)

## 🚀 Ways to Use the Application

### 1. Access the Live Demo
- Visit [RetroTech Auto Analytics](https://retrotech-auto-analytics.onrender.com)
- Log in with demo credentials
- Explore the features in read-only mode

### 2. Run Locally with Docker (Recommended)
```bash
# Clone the repository
git clone https://github.com/aevyy/RetroTech-Auto-Analytics.git
cd RetroTech-Auto-Analytics

# Start with Docker Compose
docker-compose up
```
Then visit http://localhost:5000 in your browser

### 3. Run Locally without Docker
```bash
# Clone the repository
git clone https://github.com/aevyy/RetroTech-Auto-Analytics.git
cd RetroTech-Auto-Analytics

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```
Then visit http://localhost:5000 in your browser

## 📱 Using the Dashboard

1. **Vehicle Configuration**
   - Enter your vehicle details in the configuration panel
   - Supported fields: Make, Model, Year, Engine Type, Transmission

2. **Real-time Monitoring**
   - Engine Status: Temperature, RPM, Oil Level, Coolant
   - Performance Metrics: Speed, Fuel Level, Range, Efficiency
   - System Status: OBD Connection, Data Collection, Updates

3. **Analytics Features**
   - Performance History
   - Driving Patterns Analysis
   - Maintenance Predictions
   - Health Scoring

4. **Maintenance Management**
   - View Next Service Date
   - Check Component Health
   - Monitor Battery Status
   - Track Oil Life

## 🔌 Connecting Real OBD-II Device

1. **Hardware Requirements**
   - ELM327 OBD-II adapter (Bluetooth or USB)
   - Compatible vehicle (1996 or newer)

2. **Connection Steps**
   - Plug the OBD-II adapter into your vehicle's port
   - Connect to the adapter via Bluetooth/USB
   - Enter the connection details in the app settings

3. **Demo Mode**
   - If no OBD-II device is connected, the app runs in demo mode
   - Demo mode uses simulated data for testing

## 🛠️ Troubleshooting

Common issues and solutions:
1. **Can't Connect to OBD**
   - Check if adapter is properly plugged in
   - Verify Bluetooth/USB connection
   - Ensure vehicle ignition is on

2. **Data Not Updating**
   - Check internet connection
   - Refresh the browser
   - Clear browser cache

3. **Charts Not Loading**
   - Enable JavaScript in browser
   - Try a different browser
   - Check console for errors

## 📞 Support

Need help? Contact us:
- 📧 Email: support@retrotech-analytics.com
- 💬 Discord: [Join our community](https://discord.gg/retrotech)
- 🐦 Twitter: [@RetroTechAuto](https://twitter.com/RetroTechAuto)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔮 Future Enhancements

- Integration with real vehicle OBD-II data
- Advanced anomaly detection
- Predictive maintenance ML models
- Market value prediction
- Parts inventory management
- Historical trend analysis

## 🎨 UI/UX Features

- Synthwave-inspired design
- CRT screen effect
- Real-time data updates
- Responsive layout
- Interactive dashboards

## 🌟 Portfolio Value

This project demonstrates expertise in:
- Full-stack development
- Data science and ML implementation
- Real-time data processing
- Modern UI/UX design
- Automotive systems integration
- Computational intelligence applications 