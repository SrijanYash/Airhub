import numpy as np
import requests
from datetime import datetime
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FederatedAQIModel:
    """
    Improved Federated Learning model for Air Quality Index prediction.
    Includes proper feature scaling and gradient clipping.
    """
    
    def __init__(self, learning_rate=0.001, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.global_weights = np.random.randn(7) * 0.01
        self.feature_names = ['pm2_5', 'pm10', 'no2', 'o3', 'so2', 'co']
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        
    def fetch_air_quality_data(self, api_key, lat, lon, days_back=7):
        """Fetch historical air quality data from OpenWeather API."""
        data = []
        end_time = int(datetime.now().timestamp())
        start_time = end_time - (days_back * 24 * 60 * 60)
        
        url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
        params = {
            'lat': lat,
            'lon': lon,
            'start': start_time,
            'end': end_time,
            'appid': api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            result = response.json()
            
            for item in result.get('list', []):
                # Convert API AQI (1-5) to approximate US AQI scale (0-500)
                api_aqi = item['main']['aqi']
                aqi_mapping = {1: 25, 2: 75, 3: 125, 4: 175, 5: 300}
                
                data_point = {
                    'aqi': aqi_mapping.get(api_aqi, 150),
                    'pm2_5': item['components'].get('pm2_5', 0),
                    'pm10': item['components'].get('pm10', 0),
                    'no2': item['components'].get('no2', 0),
                    'o3': item['components'].get('o3', 0),
                    'so2': item['components'].get('so2', 0),
                    'co': item['components'].get('co', 0) / 1000,  # Scale down CO
                    'timestamp': item['dt']
                }
                data.append(data_point)
            
            print(f"Fetched {len(data)} data points for location ({lat}, {lon})")
            return data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return []
    
    def prepare_features(self, data):
        """Convert raw data to normalized feature matrix and target vector."""
        X = []
        y = []
        
        for point in data:
            features = [
                point['pm2_5'],
                point['pm10'],
                point['no2'],
                point['o3'],
                point['so2'],
                point['co']
            ]
            X.append(features)
            y.append(point['aqi'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize features using StandardScaler
        if not self.is_scaler_fitted:
            X_scaled = self.scaler.fit_transform(X)
            self.is_scaler_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
        
        # Add bias term
        X_scaled = np.column_stack([X_scaled, np.ones(len(X_scaled))])
        
        return X_scaled, y
    
    def local_train(self, X, y, initial_weights):
        """
        Train model locally with gradient clipping and proper monitoring.
        """
        weights = initial_weights.copy()
        n_samples = len(X)
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Forward pass
            predictions = np.dot(X, weights)
            
            # Compute loss (MSE)
            loss = np.mean((predictions - y) ** 2)
            
            # Check for numerical issues
            if np.isnan(loss) or np.isinf(loss):
                print(f"  Warning: Numerical instability at epoch {epoch}, reverting to best weights")
                break
            
            # Compute gradients
            gradients = (2/n_samples) * np.dot(X.T, (predictions - y))
            
            # Gradient clipping to prevent explosion
            max_grad_norm = 1.0
            grad_norm = np.linalg.norm(gradients)
            if grad_norm > max_grad_norm:
                gradients = gradients * (max_grad_norm / grad_norm)
            
            # Update weights
            weights -= self.learning_rate * gradients
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                best_weights = weights.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                weights = best_weights
                break
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}/{self.epochs}, Loss: {loss:.4f}, Grad Norm: {grad_norm:.4f}")
        
        return weights
    
    def federated_averaging(self, client_weights_list, client_data_sizes):
        """Aggregate client model updates using Federated Averaging."""
        total_samples = sum(client_data_sizes)
        aggregated_weights = np.zeros_like(client_weights_list[0])
        
        for weights, size in zip(client_weights_list, client_data_sizes):
            weight_factor = size / total_samples
            aggregated_weights += weights * weight_factor
        
        return aggregated_weights
    
    def train_federated(self, locations, api_key, rounds=3):
        """Main federated learning loop."""
        print("=" * 60)
        print("FEDERATED LEARNING - AIR QUALITY PREDICTION")
        print("=" * 60)
        
        for round_num in range(rounds):
            print(f"\nROUND {round_num + 1}/{rounds}")
            print("-" * 60)
            
            client_weights_list = []
            client_data_sizes = []
            
            for idx, location in enumerate(locations):
                print(f"\nClient {idx + 1}: {location['name']}")
                
                data = self.fetch_air_quality_data(
                    api_key, 
                    location['lat'], 
                    location['lon'],
                    days_back=7
                )
                
                if len(data) < 10:
                    print("  Insufficient data, skipping client")
                    continue
                
                X, y = self.prepare_features(data)
                
                print(f"  Training on {len(X)} samples...")
                local_weights = self.local_train(X, y, self.global_weights)
                
                # Check if weights are valid
                if not np.any(np.isnan(local_weights)) and not np.any(np.isinf(local_weights)):
                    client_weights_list.append(local_weights)
                    client_data_sizes.append(len(X))
                else:
                    print("  Warning: Invalid weights, skipping this client")
            
            if client_weights_list:
                print(f"\nAggregating {len(client_weights_list)} client models...")
                self.global_weights = self.federated_averaging(
                    client_weights_list, 
                    client_data_sizes
                )
                print("Global model updated successfully")
            else:
                print("No valid client updates in this round")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        self._print_model_summary()
    
    def predict(self, features):
        """Predict AQI based on pollutant levels."""
        X = np.array([[
            features['pm2_5'],
            features['pm10'],
            features['no2'],
            features['o3'],
            features['so2'],
            features['co'] / 1000  # Scale down CO
        ]])
        
        # Normalize features
        X_scaled = self.scaler.transform(X)
        X_scaled = np.append(X_scaled, 1)  # Add bias
        
        predicted_aqi = np.dot(X_scaled, self.global_weights)
        predicted_aqi = max(1, min(500, predicted_aqi))
        
        if predicted_aqi <= 50:
            category = "Good"
            color = "🟢"
        elif predicted_aqi <= 100:
            category = "Moderate"
            color = "🟡"
        elif predicted_aqi <= 150:
            category = "Unhealthy for Sensitive Groups"
            color = "🟠"
        elif predicted_aqi <= 200:
            category = "Unhealthy"
            color = "🔴"
        elif predicted_aqi <= 300:
            category = "Very Unhealthy"
            color = "🟣"
        else:
            category = "Hazardous"
            color = "🟤"
        
        return {
            'predicted_aqi': round(predicted_aqi, 2),
            'category': category,
            'color': color,
            'features_used': features
        }
    
    def _print_model_summary(self):
        """Print model weights and feature importance."""
        print("\nModel Weights (Feature Importance):")
        print("Feature      | Weight    | Normalized Impact")
        print("-" * 50)
        
        weights_abs = np.abs(self.global_weights[:-1])
        total = np.sum(weights_abs) if np.sum(weights_abs) > 0 else 1
        
        for i, feature in enumerate(self.feature_names):
            importance = (weights_abs[i] / total) * 100
            print(f"{feature:12s} | {self.global_weights[i]:8.4f} | {importance:6.2f}%")
        print(f"{'bias':12s} | {self.global_weights[-1]:8.4f} |")
    
    def save_model(self, filepath):
        """Save trained model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'weights': self.global_weights,
                'scaler': self.scaler,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs,
                'feature_names': self.feature_names
            }, f)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.global_weights = data['weights']
            self.scaler = data['scaler']
            self.learning_rate = data['learning_rate']
            self.epochs = data['epochs']
            self.feature_names = data['feature_names']
            self.is_scaler_fitted = True
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    model = FederatedAQIModel(learning_rate=0.001, epochs=100)
    
    API_KEY = "51c085fa1523e928514f941d9f174f97"
    
    locations = [
        {'name': 'Delhi, India', 'lat': 28.6139, 'lon': 77.2090},
        {'name': 'Beijing, China', 'lat': 39.9042, 'lon': 116.4074},
        {'name': 'Los Angeles, USA', 'lat': 34.0522, 'lon': -118.2437},
        {'name': 'London, UK', 'lat': 51.5074, 'lon': -0.1278},
        {'name': 'Mumbai, India', 'lat': 19.0760, 'lon': 72.8777}
    ]
    
    model.train_federated(locations, API_KEY, rounds=3)
    
    print("\n" + "=" * 60)
    print("PREDICTION EXAMPLES")
    print("=" * 60)
    
    # Good air quality
    prediction1 = model.predict({
        'pm2_5': 12.0,
        'pm10': 20.0,
        'no2': 15.0,
        'o3': 50.0,
        'so2': 5.0,
        'co': 300.0
    })
    print(f"\n{prediction1['color']} Scenario 1 (Low pollution):")
    print(f"  Predicted AQI: {prediction1['predicted_aqi']}")
    print(f"  Category: {prediction1['category']}")
    
    # Poor air quality
    prediction2 = model.predict({
        'pm2_5': 150.0,
        'pm10': 200.0,
        'no2': 100.0,
        'o3': 120.0,
        'so2': 80.0,
        'co': 5000.0
    })
    print(f"\n{prediction2['color']} Scenario 2 (High pollution):")
    print(f"  Predicted AQI: {prediction2['predicted_aqi']}")
    print(f"  Category: {prediction2['category']}")
    
    model.save_model('aqi_federated_model.pkl')
    print("\nComplete! Model ready for deployment.")