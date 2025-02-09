from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objs as go
import plotly.utils
import json
from vehicle_data import VehicleDataCollector

app = Flask(__name__)
CORS(app)

# Initialize vehicle data collector
vehicle_data = VehicleDataCollector()

def create_gauge_chart(value, title, min_val=0, max_val=100):
    """Create a gauge chart using plotly"""
    try:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),  # Ensure value is float
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'color': "#00ff00", 'family': "VT323"}},
            number={'font': {'color': "#00ff00", 'family': "VT323"}},
            gauge={
                'axis': {'range': [min_val, max_val], 'tickfont': {'color': "#00ff00", 'family': "VT323"}},
                'bar': {'color': "#00ff00"},
                'bgcolor': "black",
                'borderwidth': 2,
                'bordercolor': "#00ff00",
                'steps': [
                    {'range': [min_val, max_val*0.3], 'color': 'rgba(255, 0, 0, 0.2)'},
                    {'range': [max_val*0.3, max_val*0.7], 'color': 'rgba(255, 255, 0, 0.2)'},
                    {'range': [max_val*0.7, max_val], 'color': 'rgba(0, 255, 0, 0.2)'}
                ],
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#00ff00", 'family': "VT323"},
            margin=dict(l=20, r=20, t=50, b=20),
            height=250
        )
        
        return json.loads(fig.to_json())
    except Exception as e:
        print(f"Error creating gauge chart: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/system-status')
def system_status():
    """Get the status of various system components"""
    return jsonify({
        'obd_connected': vehicle_data.connection is not None and vehicle_data.connection.is_connected(),
        'ml_models_ready': vehicle_data.anomaly_detector is not None and vehicle_data.maintenance_predictor is not None,
        'last_update': datetime.now().isoformat()
    })

@app.route('/api/vehicle-stats')
def vehicle_stats():
    try:
        data = vehicle_data.get_real_time_data()
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/maintenance-prediction')
def maintenance_prediction():
    try:
        prediction = vehicle_data.predict_maintenance()
        return jsonify({
            'success': True,
            'data': prediction
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/market-analysis')
def market_analysis():
    try:
        # Example vehicle details - you would typically get these from user input
        market_data = vehicle_data.get_market_data('Toyota', 'Camry', 2020)
        return jsonify({
            'success': True,
            'data': market_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/charts/engine-health')
def engine_health_chart():
    try:
        data = vehicle_data.get_real_time_data()
        chart_data = create_gauge_chart(
            data['engine_health'],
            'Engine Health Score',
            0,
            100
        )
        if chart_data is None:
            raise Exception("Failed to create engine health chart")
        return jsonify(chart_data)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/charts/fuel-efficiency')
def fuel_efficiency_chart():
    try:
        data = vehicle_data.get_real_time_data()
        chart_data = create_gauge_chart(
            data['fuel_efficiency'],
            'Fuel Efficiency (MPG)',
            0,
            50
        )
        if chart_data is None:
            raise Exception("Failed to create fuel efficiency chart")
        return jsonify(chart_data)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/critical-alerts')
def critical_alerts():
    try:
        data = vehicle_data.get_real_time_data()
        critical_components = vehicle_data.identify_critical_components(data)
        return jsonify({
            'success': True,
            'data': {
                'alerts': critical_components,
                'total_alerts': len(critical_components)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/maintenance-costs')
def maintenance_costs():
    try:
        data = vehicle_data.get_real_time_data()
        costs = vehicle_data.estimate_maintenance_costs(data)
        return jsonify({
            'success': True,
            'data': costs
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/historical-trends')
def historical_trends():
    """Get historical trend analysis"""
    try:
        analysis = vehicle_data.get_historical_analysis()
        if not analysis:
            return jsonify({
                'success': False,
                'error': 'No historical data available'
            }), 404
        return jsonify({
            'success': True,
            'data': analysis
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/driving-analytics')
def driving_analytics():
    """Get driving pattern analytics"""
    try:
        data = vehicle_data.get_real_time_data()
        return jsonify({
            'success': True,
            'data': {
                'current_pattern': data.get('driving_pattern', 'unknown'),
                'performance_score': data.get('performance_score', 0),
                'efficiency_score': data.get('efficiency_score', 0)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/set-vehicle-config', methods=['POST'])
def set_vehicle_config():
    """Set vehicle configuration"""
    try:
        config = request.json
        vehicle_data.set_vehicle_config(
            make=config.get('make'),
            model=config.get('model'),
            year=config.get('year'),
            engine_type=config.get('engine_type', 'gasoline'),
            transmission=config.get('transmission', 'automatic')
        )
        return jsonify({
            'success': True,
            'message': 'Vehicle configuration updated successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/charts/performance-history')
def performance_history_chart():
    """Get historical performance chart data"""
    try:
        if vehicle_data.historical_data.empty:
            raise Exception("No historical data available")
            
        df = vehicle_data.historical_data.tail(50)
        
        fig = go.Figure()
        
        # Add performance metrics
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['engine_health'],
            name='Engine Health',
            line=dict(color='#00ff00', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['fuel_efficiency'],
            name='Fuel Efficiency',
            line=dict(color='#00ffff', width=2)
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#00ff00", 'family': "VT323"},
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,255,0,0.1)',
                title='Time',
                tickfont={'color': "#00ff00", 'family': "VT323"},
                title_font={'color': "#00ff00", 'family': "VT323"}
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,255,0,0.1)',
                title='Value',
                tickfont={'color': "#00ff00", 'family': "VT323"},
                title_font={'color': "#00ff00", 'family': "VT323"}
            ),
            legend=dict(
                font={'color': "#00ff00", 'family': "VT323"},
                bgcolor='rgba(0,0,0,0.5)'
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            height=300
        )
        
        return jsonify(json.loads(fig.to_json()))
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/charts/driving-patterns')
def driving_patterns_chart():
    """Get driving patterns distribution chart"""
    try:
        if vehicle_data.historical_data.empty:
            raise Exception("No historical data available")
            
        pattern_counts = vehicle_data.historical_data['driving_pattern'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=pattern_counts.index,
            values=pattern_counts.values,
            hole=.3,
            textfont={'color': "#00ff00", 'family': "VT323"},
            marker=dict(
                colors=['#00ff00', '#ffff00', '#ff0000'],
                line=dict(color='#000000', width=2)
            )
        )])
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#00ff00", 'family': "VT323"},
            margin=dict(l=20, r=20, t=50, b=20),
            height=300,
            showlegend=True,
            legend=dict(
                font={'color': "#00ff00", 'family': "VT323"},
                bgcolor='rgba(0,0,0,0.5)'
            )
        )
        
        return jsonify(json.loads(fig.to_json()))
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Try to connect to OBD-II
    if vehicle_data.connect_obd():
        print("Successfully connected to OBD-II interface")
    else:
        print("No OBD-II connection available, using mock data")
    
    app.run(debug=True)
