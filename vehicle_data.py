import pandas as pd
import numpy as np
from datetime import datetime
import requests
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

class VehicleConfig:
    def __init__(self, make, model, year, engine_type, transmission):
        self.make = make
        self.model = model
        self.year = year
        self.engine_type = engine_type
        self.transmission = transmission
        
        # Vehicle-specific parameters
        self.parameters = self.get_vehicle_parameters()
    
    def get_vehicle_parameters(self):
        """Get vehicle-specific parameters based on configuration"""
        # These would typically come from a database, using mock data for demonstration
        base_params = {
            'idle_rpm': 800,
            'max_rpm': 6500,
            'optimal_rpm': 1500,
            'optimal_temp': 90,
            'max_temp': 120,
            'optimal_oil_temp': 100,
            'max_oil_temp': 130,
            'base_fuel_efficiency': 25,
        }
        
        # Adjust parameters based on engine type
        if self.engine_type == 'hybrid':
            base_params.update({
                'idle_rpm': 0,
                'optimal_rpm': 1200,
                'base_fuel_efficiency': 45
            })
        elif self.engine_type == 'diesel':
            base_params.update({
                'idle_rpm': 700,
                'optimal_rpm': 2000,
                'optimal_temp': 85,
                'base_fuel_efficiency': 30
            })
        
        # Adjust for vehicle age
        age = datetime.now().year - self.year
        base_params['base_fuel_efficiency'] *= max(0.7, 1 - (age * 0.02))
        
        return base_params

class VehicleDataCollector:
    def __init__(self):
        self.connection = None
        self.market_data_cache = {}
        self.scaler = StandardScaler()
        self.anomaly_detector = None
        self.maintenance_predictor = None
        self.price_predictor = None
        self.driving_pattern_classifier = None
        self.vehicle_config = None
        self.historical_data = pd.DataFrame()
        self.initialize_ml_models()
    
    def set_vehicle_config(self, make, model, year, engine_type='gasoline', transmission='automatic'):
        """Set vehicle configuration"""
        self.vehicle_config = VehicleConfig(make, model, year, engine_type, transmission)
        # Retrain models with vehicle-specific parameters
        self.initialize_ml_models()

    def initialize_ml_models(self):
        """Initialize and train ML models with historical data"""
        # Generate initial training data
        historical_data = self.generate_mock_historical_data()
        self.historical_data = historical_data
        
        # Prepare data for anomaly detection
        feature_columns = ['engine_temp', 'oil_temp', 'rpm']
        X_train = historical_data[feature_columns].values
        
        # Train anomaly detector
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_detector.fit(X_train)
        
        # Prepare data for maintenance prediction
        y_train = historical_data['maintenance_performed'].values
        self.maintenance_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.maintenance_predictor.fit(X_train, y_train)
        
        # Train price predictor
        self.price_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        price_features = self.generate_price_training_data()
        self.price_predictor.fit(
            price_features[['year', 'mileage', 'condition']],
            price_features['price']
        )
        
        # Train driving pattern classifier
        self.driving_pattern_classifier = KMeans(n_clusters=3, random_state=42)
        pattern_features = historical_data[['rpm', 'speed', 'throttle_pos']].values
        self.driving_pattern_classifier.fit(pattern_features)
        
        # Fit scaler
        self.scaler.fit(X_train)

    def connect_obd(self):
        """Mock connection for demonstration"""
        print("Using mock data for demonstration")
        return True

    def get_real_time_data(self):
        """Get mock vehicle data"""
        data = self.get_mock_data()
        return self.process_real_time_data(data)

    def get_mock_data(self):
        """Generate realistic mock data based on vehicle configuration"""
        if not self.vehicle_config:
            # Default parameters when no vehicle config is set
            base_temp = 90 + np.sin(datetime.now().timestamp() / 300) * 5
            base_rpm = 1500 + np.sin(datetime.now().timestamp() / 60) * 300
            
            return {
                'engine_temp': base_temp + np.random.normal(0, 2),
                'oil_temp': base_temp + 10 + np.random.normal(0, 3),
                'rpm': base_rpm + np.random.normal(0, 100),
                'speed': max(0, (base_rpm - 800) / 50 + np.random.normal(0, 5)),
                'throttle_pos': max(0, min(100, 20 + np.sin(datetime.now().timestamp() / 120) * 15)),
                'fuel_level': max(0, min(100, 75 + np.random.normal(0, 1))),
                'timing_advance': 12 + np.random.normal(0, 1),
                'maf': 15 + np.random.normal(0, 1),
                'o2_voltage': max(0, min(1.1, 0.9 + np.random.normal(0, 0.05)))
            }
        
        # Vehicle config is set, use its parameters
        params = self.vehicle_config.parameters
        base_temp = params['optimal_temp'] + np.sin(datetime.now().timestamp() / 300) * 5
        base_rpm = params['optimal_rpm'] + np.sin(datetime.now().timestamp() / 60) * 300
        
        return {
            'engine_temp': base_temp + np.random.normal(0, 2),
            'oil_temp': base_temp + 10 + np.random.normal(0, 3),
            'rpm': base_rpm + np.random.normal(0, 100),
            'speed': max(0, (base_rpm - params['idle_rpm']) / 50 + np.random.normal(0, 5)),
            'throttle_pos': max(0, min(100, 20 + np.sin(datetime.now().timestamp() / 120) * 15)),
            'fuel_level': max(0, min(100, 75 + np.random.normal(0, 1))),
            'timing_advance': 12 + np.random.normal(0, 1),
            'maf': 15 + np.random.normal(0, 1),
            'o2_voltage': max(0, min(1.1, 0.9 + np.random.normal(0, 0.05)))
        }

    def process_real_time_data(self, data):
        """Process and analyze real-time data with enhanced features"""
        try:
            # Basic processing
            feature_values = [
                data['engine_temp'],
                data['oil_temp'],
                data['rpm']
            ]
            data_array = np.array([feature_values])
            data_scaled = self.scaler.transform(data_array)
            
            # Enhanced analytics
            data['anomaly_score'] = float(self.anomaly_detector.score_samples(data_scaled)[0])
            data['fuel_efficiency'] = self.calculate_fuel_efficiency(data)
            data['engine_health'] = self.calculate_engine_health(data)
            
            # Analyze driving pattern
            pattern_features = np.array([[
                data['rpm'],
                data['speed'],
                data['throttle_pos']
            ]])
            data['driving_pattern'] = self.classify_driving_pattern(pattern_features)
            
            # Calculate performance metrics
            data['performance_score'] = self.calculate_performance_score(data)
            data['efficiency_score'] = self.calculate_efficiency_score(data)
            
            # Ensure all required fields are present
            if 'engine_health' not in data:
                data['engine_health'] = 50.0  # Default value
            if 'driving_pattern' not in data:
                data['driving_pattern'] = 'normal'  # Default value
            
            # Update historical trends
            self.update_historical_trends(data)
            
            return data
        except Exception as e:
            print(f"Error processing real-time data: {str(e)}")
            # Return data with default values if processing fails
            data.update({
                'engine_health': 50.0,
                'fuel_efficiency': 25.0,
                'driving_pattern': 'normal',
                'performance_score': 50.0,
                'efficiency_score': 50.0,
                'anomaly_score': 0.0
            })
            return data

    def classify_driving_pattern(self, features):
        """Classify driving pattern into categories"""
        pattern_id = self.driving_pattern_classifier.predict(features)[0]
        patterns = ['economic', 'normal', 'aggressive']
        return patterns[pattern_id]

    def calculate_performance_score(self, data):
        """Calculate overall performance score"""
        if not self.vehicle_config:
            return 50.0
            
        params = self.vehicle_config.parameters
        
        # Calculate various performance factors
        rpm_efficiency = 1.0 - abs(data['rpm'] - params['optimal_rpm']) / params['max_rpm']
        temp_efficiency = 1.0 - abs(data['engine_temp'] - params['optimal_temp']) / params['max_temp']
        throttle_response = data['throttle_pos'] / 100.0
        
        # Combine factors with weights
        score = (rpm_efficiency * 0.4 + temp_efficiency * 0.3 + throttle_response * 0.3) * 100
        return max(0, min(100, score))

    def calculate_efficiency_score(self, data):
        """Calculate efficiency score based on multiple factors"""
        if not self.vehicle_config:
            return 50.0
            
        params = self.vehicle_config.parameters
        
        # Calculate efficiency factors
        fuel_factor = data['fuel_efficiency'] / params['base_fuel_efficiency']
        temp_factor = 1.0 - abs(data['engine_temp'] - params['optimal_temp']) / params['max_temp']
        rpm_factor = 1.0 - abs(data['rpm'] - params['optimal_rpm']) / params['max_rpm']
        
        # Combine factors with weights
        score = (fuel_factor * 0.4 + temp_factor * 0.3 + rpm_factor * 0.3) * 100
        return max(0, min(100, score))

    def update_historical_trends(self, data):
        """Update historical data with new measurements"""
        new_data = pd.DataFrame([{
            'timestamp': datetime.now(),
            'engine_temp': data['engine_temp'],
            'oil_temp': data['oil_temp'],
            'rpm': data['rpm'],
            'speed': data['speed'],
            'throttle_pos': data['throttle_pos'],
            'fuel_efficiency': data['fuel_efficiency'],
            'engine_health': data['engine_health'],
            'driving_pattern': data['driving_pattern']
        }])
        
        self.historical_data = pd.concat([self.historical_data, new_data]).tail(1000)

    def get_historical_analysis(self):
        """Get analysis of historical data"""
        if self.historical_data.empty:
            return None
            
        return {
            'efficiency_trend': self.calculate_trend(self.historical_data['fuel_efficiency']),
            'health_trend': self.calculate_trend(self.historical_data['engine_health']),
            'driving_patterns': self.historical_data['driving_pattern'].value_counts().to_dict(),
            'avg_efficiency': self.historical_data['fuel_efficiency'].mean(),
            'avg_health': self.historical_data['engine_health'].mean()
        }

    def calculate_trend(self, series):
        """Calculate trend direction and magnitude"""
        if len(series) < 2:
            return {'direction': 'stable', 'magnitude': 0}
            
        slope = np.polyfit(range(len(series)), series, 1)[0]
        
        if abs(slope) < 0.01:
            direction = 'stable'
        else:
            direction = 'improving' if slope > 0 else 'degrading'
            
        return {
            'direction': direction,
            'magnitude': abs(slope)
        }

    def generate_price_training_data(self):
        """Generate mock training data for price prediction"""
        n_samples = 1000
        years = np.random.randint(2010, 2024, n_samples)
        mileages = np.random.normal(12000 * (2024 - years), 5000)
        conditions = np.random.normal(90 - (2024 - years) * 2, 5)
        
        # Base price calculation
        base_prices = 30000 * np.exp(-0.1 * (2024 - years))
        # Adjust for mileage and condition
        prices = base_prices * (1 - mileages / 200000) * (conditions / 100)
        
        return pd.DataFrame({
            'year': years,
            'mileage': mileages,
            'condition': conditions,
            'price': prices
        })

    def calculate_fuel_efficiency(self, data):
        """Calculate mock fuel efficiency"""
        base_efficiency = 25  # Base MPG
        
        # Factors that affect efficiency
        rpm_factor = 1.0 - abs(data['rpm'] - 1500) / 3000  # Optimal RPM around 1500
        throttle_factor = 1.0 - data['throttle_pos'] / 200  # Higher throttle = lower efficiency
        temp_factor = 1.0 - abs(data['engine_temp'] - 90) / 100  # Optimal temp around 90
        
        efficiency = base_efficiency * rpm_factor * throttle_factor * temp_factor
        return max(15, min(35, efficiency + np.random.normal(0, 1)))

    def calculate_engine_health(self, data):
        """Calculate engine health score based on multiple parameters"""
        weights = {
            'engine_temp': 0.2,
            'oil_temp': 0.2,
            'rpm': 0.15,
            'throttle_pos': 0.15,
            'timing_advance': 0.15,
            'o2_voltage': 0.15
        }
        
        health_score = 100
        for param, weight in weights.items():
            if param in data:
                # Define normal ranges for each parameter
                ranges = {
                    'engine_temp': (75, 105),
                    'oil_temp': (82, 115),
                    'rpm': (600, 2500),
                    'throttle_pos': (0, 100),
                    'timing_advance': (5, 20),
                    'o2_voltage': (0.6, 1.1)
                }
                
                min_val, max_val = ranges[param]
                if data[param] < min_val or data[param] > max_val:
                    health_score -= (weight * 100)
        
        return max(0, min(100, health_score))

    def get_market_data(self, make, model, year):
        """Generate mock market data"""
        cache_key = f"{make}_{model}_{year}"
        if cache_key in self.market_data_cache:
            return self.market_data_cache[cache_key]

        # Generate realistic mock market data
        base_value = 25000
        age_factor = max(0.4, 1 - (datetime.now().year - year) * 0.08)
        market_condition = np.random.normal(1, 0.1)
        current_value = base_value * age_factor * market_condition

        market_data = {
            'current_value': current_value,
            'market_trend': np.random.choice(['up', 'down', 'stable'], p=[0.3, 0.3, 0.4]),
            'similar_listings': np.random.randint(10, 100),
            'price_prediction': current_value * (1 + np.random.normal(0, 0.05))
        }

        self.market_data_cache[cache_key] = market_data
        return market_data

    def predict_maintenance(self, historical_data=None):
        """Predict maintenance needs based on current and historical data"""
        if historical_data is None:
            historical_data = self.generate_mock_historical_data()

        current_data = self.get_real_time_data()
        
        # Prepare features for prediction
        features = np.array([[
            current_data['engine_temp'],
            current_data['oil_temp'],
            current_data['rpm'],
            current_data['timing_advance'],
            current_data.get('anomaly_score', 0)
        ]])
        
        # Make predictions
        maintenance_score = self.maintenance_predictor.predict(features)[0]
        
        return {
            'maintenance_score': maintenance_score,
            'next_service_days': self.predict_service_interval(maintenance_score),
            'critical_components': self.identify_critical_components(current_data),
            'estimated_costs': self.estimate_maintenance_costs(current_data)
        }

    def generate_mock_historical_data(self):
        """Generate mock historical data for training"""
        n_samples = 1000
        timestamps = pd.date_range(end=datetime.now(), periods=n_samples, freq='1min')
        
        # Generate base values with some randomness
        base_temp = 90 + np.sin(np.linspace(0, 10, n_samples)) * 5
        base_rpm = 1500 + np.sin(np.linspace(0, 20, n_samples)) * 300
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'engine_temp': base_temp + np.random.normal(0, 2, n_samples),
            'oil_temp': base_temp + 10 + np.random.normal(0, 3, n_samples),
            'rpm': base_rpm + np.random.normal(0, 100, n_samples),
            'speed': (base_rpm - 800) / 50 + np.random.normal(0, 5, n_samples),
            'throttle_pos': 20 + np.sin(np.linspace(0, 15, n_samples)) * 15 + np.random.normal(0, 2, n_samples),
            'fuel_level': 75 + np.random.normal(0, 1, n_samples),
            'timing_advance': 12 + np.random.normal(0, 1, n_samples),
            'maf': 15 + np.random.normal(0, 1, n_samples),
            'o2_voltage': 0.9 + np.random.normal(0, 0.05, n_samples),
            'maintenance_performed': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
        })
        
        # Ensure values are within realistic ranges
        data['speed'] = data['speed'].clip(0, 200)
        data['throttle_pos'] = data['throttle_pos'].clip(0, 100)
        data['fuel_level'] = data['fuel_level'].clip(0, 100)
        data['o2_voltage'] = data['o2_voltage'].clip(0, 1.1)
        
        return data

    def predict_service_interval(self, maintenance_score):
        """Predict days until next service is needed"""
        if maintenance_score > 90:
            return np.random.randint(60, 90)
        elif maintenance_score > 70:
            return np.random.randint(30, 60)
        else:
            return np.random.randint(0, 30)

    def identify_critical_components(self, data):
        """Identify components that need attention"""
        critical_components = []
        
        thresholds = {
            'engine_temp': (110, 'Engine Temperature High'),
            'oil_temp': (120, 'Oil Temperature High'),
            'timing_advance': (25, 'Timing Advance Issue'),
            'o2_voltage': (1.2, 'O2 Sensor Issue')
        }
        
        for component, (threshold, message) in thresholds.items():
            if component in data and data[component] > threshold:
                critical_components.append({
                    'component': component,
                    'message': message,
                    'urgency': 'high' if data[component] > threshold * 1.1 else 'medium'
                })
        
        return critical_components

    def estimate_maintenance_costs(self, data):
        """Estimate maintenance costs based on vehicle condition"""
        base_costs = {
            'oil_change': 50,
            'air_filter': 20,
            'brake_pads': 150,
            'timing_belt': 500,
            'spark_plugs': 100
        }
        
        estimated_costs = {}
        wear_factor = (100 - self.calculate_engine_health(data)) / 100
        
        for service, base_cost in base_costs.items():
            estimated_costs[service] = base_cost * (1 + wear_factor)
        
        return estimated_costs 