"""
Multi-Step AQI Prediction Model using LSTM
Predicts AQI for next 8-12 hours
Data: Delhi AQI from Nov 1-8, 2025 (168 hourly samples)
Features: pm25, pm10, o3, no2, so2, co, no, nh3
Target: aqi for next 8-12 hours
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

class MultiStepAQIDataProcessor:
    """Process and prepare AQI data for multi-step LSTM forecasting."""
    
    def __init__(
        self, 
        csv_path: str = 'data/datasets/sample_aqi_data.csv',  # Updated default path
        forecast_horizon: int = 12  # Number of hours to predict
    ):
        self.csv_path = csv_path
        self.forecast_horizon = forecast_horizon
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        # Define features (8 pollutants)
        self.feature_columns = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'no', 'nh3']
        self.target_column = 'aqi'
        
        self.df = None
        self.X_scaled = None
        self.y_scaled = None
    
    def calculate_us_aqi_from_pm25(self, pm25: float) -> float:
        """
        Calculate US EPA AQI (0-500 scale) from PM2.5 concentration.
        Based on official EPA breakpoints.
        """
        # EPA AQI Breakpoints for PM2.5 (24-hour average)
        # Format: (C_low, C_high, I_low, I_high)
        breakpoints = [
            (0.0, 12.0, 0, 50),      # Good
            (12.1, 35.4, 51, 100),   # Moderate
            (35.5, 55.4, 101, 150),  # Unhealthy for Sensitive Groups
            (55.5, 150.4, 151, 200), # Unhealthy
            (150.5, 250.4, 201, 300),# Very Unhealthy
            (250.5, 350.4, 301, 400),# Hazardous
            (350.5, 500.4, 401, 500) # Hazardous
        ]
        
        for C_low, C_high, I_low, I_high in breakpoints:
            if C_low <= pm25 <= C_high:
                # Linear interpolation formula
                aqi = ((I_high - I_low) / (C_high - C_low)) * (pm25 - C_low) + I_low
                return round(aqi)
        
        # If PM2.5 exceeds highest breakpoint
        if pm25 > 500.4:
            return 500
        return 0
    
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load CSV and perform initial cleaning."""
        print("📁 Loading data...")
        
        # Load data
        df = pd.read_csv(self.csv_path)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Check for missing values
        missing = df[self.feature_columns + [self.target_column]].isnull().sum()
        if missing.any():
            print(f"⚠️  Missing values found:\n{missing[missing > 0]}")
            df = df.dropna(subset=self.feature_columns + [self.target_column])
        
        # Convert AQI from 0-5 scale to proper 0-500 US EPA AQI scale
        print("\n🔄 Converting AQI to standard 0-500 scale...")
        print(f"   Original AQI range: {df['aqi'].min():.1f} to {df['aqi'].max():.1f}")
        
        # Calculate proper AQI from PM2.5 values
        df['aqi'] = df['pm25'].apply(self.calculate_us_aqi_from_pm25)
        
        print(f"   Converted AQI range: {df['aqi'].min():.0f} to {df['aqi'].max():.0f}")
        print(f"   Mean AQI: {df['aqi'].mean():.1f}")
        
        print(f"✅ Loaded {len(df)} samples")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        self.df = df  # ✅ FIXED: Store dataframe to self.df
        return df
    
    def scale_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize features and target to [0, 1]."""
        print("\n🔄 Scaling data...")
        
        # Extract features and target
        X = self.df[self.feature_columns].values
        y = self.df[self.target_column].values.reshape(-1, 1)
        
        # Scale
        self.X_scaled = self.feature_scaler.fit_transform(X)
        self.y_scaled = self.target_scaler.fit_transform(y).flatten()
        
        print(f"   Features shape: {self.X_scaled.shape}")
        print(f"   Target shape: {self.y_scaled.shape}")
        
        return self.X_scaled, self.y_scaled
    
    def create_multi_step_sequences(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        sequence_length: int = 24
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for multi-step forecasting.
        
        Input: Last 'sequence_length' hours of features
        Output: Next 'forecast_horizon' hours of AQI
        """
        print(f"\n📊 Creating multi-step sequences:")
        print(f"   Input window: {sequence_length} hours")
        print(f"   Forecast horizon: {self.forecast_horizon} hours")
        
        X_seq, y_seq = [], []
        
        # Need enough data for input sequence + forecast horizon
        for i in range(len(X) - sequence_length - self.forecast_horizon + 1):
            # Input: last 'sequence_length' hours
            X_seq.append(X[i:i + sequence_length])
            
            # Target: next 'forecast_horizon' hours of AQI
            y_seq.append(y[i + sequence_length:i + sequence_length + self.forecast_horizon])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        print(f"   Created {len(X_seq)} sequences")
        print(f"   Input shape: {X_seq.shape}")    # (samples, seq_length, features)
        print(f"   Output shape: {y_seq.shape}")   # (samples, forecast_horizon)
        
        return X_seq, y_seq
    
    def prepare_data(
        self, 
        sequence_length: int = 24,
        test_size: float = 0.2
    ) -> Dict:
        """Complete data preparation pipeline."""
        
        # Load and clean
        self.load_and_clean_data()
        
        # Scale
        X_scaled, y_scaled = self.scale_data()
        
        # Create sequences
        X_seq, y_seq = self.create_multi_step_sequences(X_scaled, y_scaled, sequence_length)
        
        # Train-test split (don't shuffle time series!)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=test_size, shuffle=False
        )
        
        print(f"\n📦 Data split:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'sequence_length': sequence_length,
            'forecast_horizon': self.forecast_horizon
        }
    
    def inverse_transform_predictions(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original AQI scale."""
        original_shape = y_scaled.shape
        y_reshaped = y_scaled.reshape(-1, 1)
        y_original = self.target_scaler.inverse_transform(y_reshaped)
        return y_original.reshape(original_shape)


# ============================================================================
# 2. PYTORCH DATASET
# ============================================================================

class MultiStepAQIDataset(Dataset):
    """Custom PyTorch Dataset for multi-step AQI forecasting."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# 3. MULTI-STEP LSTM MODEL
# ============================================================================

class MultiStepAQILSTM(nn.Module):
    """LSTM model for multi-step AQI forecasting."""
    
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        forecast_horizon: int = 12,
        dropout: float = 0.2
    ):
        super(MultiStepAQILSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism (optional but helps with multi-step)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Fully connected layers for multi-step output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, forecast_horizon)  # Output: forecast_horizon predictions
        )
    
    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Apply attention to focus on important time steps
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Generate multi-step predictions
        predictions = self.fc(context)  # Shape: (batch, forecast_horizon)
        
        return predictions


# ============================================================================
# 4. TRAINING
# ============================================================================

class MultiStepAQITrainer:
    """Handles model training and evaluation for multi-step forecasting."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 150,
        learning_rate: float = 0.001
    ):
        """Train the model."""
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        print("\n🚀 Starting training...")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {learning_rate}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 25
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_multistep_aqi_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\n✅ Training complete! Best validation loss: {best_val_loss:.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_multistep_aqi_model.pth'))
    
    def evaluate(
        self,
        test_loader: DataLoader,
        target_scaler: MinMaxScaler,
        forecast_horizon: int
    ) -> Dict:
        """Evaluate model on test set."""
        
        self.model.eval()
        all_predictions = []
        all_actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch).cpu().numpy()
                all_predictions.append(pred)
                all_actuals.append(y_batch.numpy())
        
        predictions = np.vstack(all_predictions)
        actuals = np.vstack(all_actuals)
        
        # Inverse transform to original scale
        predictions_original = target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).reshape(predictions.shape)
        
        actuals_original = target_scaler.inverse_transform(
            actuals.reshape(-1, 1)
        ).reshape(actuals.shape)
        
        # Calculate metrics for each forecast step
        print("\n📊 Model Evaluation Metrics (by forecast hour):")
        print("   " + "-" * 60)
        
        hourly_metrics = []
        for h in range(forecast_horizon):
            pred_h = predictions_original[:, h]
            actual_h = actuals_original[:, h]
            
            rmse = np.sqrt(np.mean((pred_h - actual_h) ** 2))
            mae = np.mean(np.abs(pred_h - actual_h))
            mape = np.mean(np.abs((actual_h - pred_h) / actual_h)) * 100
            
            hourly_metrics.append({'hour': h+1, 'rmse': rmse, 'mae': mae, 'mape': mape})
            
            if h < 3 or h >= forecast_horizon - 3:  # Print first 3 and last 3
                print(f"   Hour +{h+1:2d}: RMSE={rmse:5.2f}, MAE={mae:5.2f}, MAPE={mape:5.2f}%")
            elif h == 3:
                print("   ...")
        
        # Overall metrics
        overall_rmse = np.sqrt(np.mean((predictions_original - actuals_original) ** 2))
        overall_mae = np.mean(np.abs(predictions_original - actuals_original))
        overall_mape = np.mean(np.abs((actuals_original - predictions_original) / actuals_original)) * 100
        
        print("   " + "-" * 60)
        print(f"   Overall: RMSE={overall_rmse:.2f}, MAE={overall_mae:.2f}, MAPE={overall_mape:.2f}%")
        
        return {
            'predictions': predictions_original,
            'actuals': actuals_original,
            'hourly_metrics': hourly_metrics,
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'overall_mape': overall_mape
        }
    
    def predict_future(
        self,
        last_sequence: np.ndarray,
        feature_scaler: MinMaxScaler,
        target_scaler: MinMaxScaler,
        forecast_horizon: int
    ) -> np.ndarray:
        """Predict AQI for next N hours given last sequence."""
        
        self.model.eval()
        
        # Scale input
        last_sequence_scaled = feature_scaler.transform(last_sequence)
        
        # Convert to tensor
        X = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions_scaled = self.model(X).cpu().numpy()
        
        # Inverse transform
        predictions = target_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        return predictions


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def plot_multi_step_results(results: Dict, trainer: MultiStepAQITrainer, forecast_horizon: int):
    """Plot training history and multi-step predictions."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Training History
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(trainer.train_losses, label='Training Loss', linewidth=2)
    ax1.plot(trainer.val_losses, label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training History', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sample Predictions (First test sample)
    ax2 = fig.add_subplot(gs[1, 0])
    sample_idx = 0
    hours = np.arange(1, forecast_horizon + 1)
    ax2.plot(hours, results['actuals'][sample_idx], 'o-', 
             label='Actual', linewidth=2, markersize=8)
    ax2.plot(hours, results['predictions'][sample_idx], 's-', 
             label='Predicted', linewidth=2, markersize=8)
    ax2.set_xlabel('Hours Ahead', fontsize=12)
    ax2.set_ylabel('AQI', fontsize=12)
    ax2.set_title(f'Sample Forecast (Test Sample #{sample_idx+1})', 
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(hours)
    
    # Plot 3: Another Sample
    ax3 = fig.add_subplot(gs[1, 1])
    sample_idx = min(5, len(results['actuals']) - 1)
    ax3.plot(hours, results['actuals'][sample_idx], 'o-', 
             label='Actual', linewidth=2, markersize=8)
    ax3.plot(hours, results['predictions'][sample_idx], 's-', 
             label='Predicted', linewidth=2, markersize=8)
    ax3.set_xlabel('Hours Ahead', fontsize=12)
    ax3.set_ylabel('AQI', fontsize=12)
    ax3.set_title(f'Sample Forecast (Test Sample #{sample_idx+1})', 
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(hours)
    
    # Plot 4: Error by Forecast Horizon
    ax4 = fig.add_subplot(gs[2, 0])
    hourly_rmse = [m['rmse'] for m in results['hourly_metrics']]
    ax4.bar(hours, hourly_rmse, alpha=0.7, color='coral')
    ax4.set_xlabel('Hours Ahead', fontsize=12)
    ax4.set_ylabel('RMSE (AQI)', fontsize=12)
    ax4.set_title('Forecast Error by Hour', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticks(hours)
    
    # Plot 5: All Predictions Scatter
    ax5 = fig.add_subplot(gs[2, 1])
    all_actuals = results['actuals'].flatten()
    all_predictions = results['predictions'].flatten()
    ax5.scatter(all_actuals, all_predictions, alpha=0.5, s=30)
    
    # Perfect prediction line
    min_val = min(all_actuals.min(), all_predictions.min())
    max_val = max(all_actuals.max(), all_predictions.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax5.set_xlabel('Actual AQI', fontsize=12)
    ax5.set_ylabel('Predicted AQI', fontsize=12)
    ax5.set_title(f'All Predictions (RMSE: {results["overall_rmse"]:.2f})', 
                  fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    plt.savefig('multi_step_aqi_predictions.png', dpi=300, bbox_inches='tight')
    print("\n📈 Results saved to 'multi_step_aqi_predictions.png'")
    plt.show()


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("="*70)
    print("🌍 MULTI-STEP AQI PREDICTION MODEL - DELHI DATA")
    print("="*70)
    
    # Configuration
    FORECAST_HORIZON = 12  # Predict next 12 hours (can change to 8)
    SEQUENCE_LENGTH = 24   # Use last 24 hours as input
    BATCH_SIZE = 8         # Smaller batch for limited data
    EPOCHS = 150
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n⚙️  Configuration:")
    print(f"   Input window: {SEQUENCE_LENGTH} hours")
    print(f"   Forecast horizon: {FORECAST_HORIZON} hours")
    print(f"   Device: {device}")
    
    # 1. Prepare data
    processor = MultiStepAQIDataProcessor(
        'data/datasets/sample_aqi_data.csv',  # Updated path for your structure
        forecast_horizon=FORECAST_HORIZON
    )
    data = processor.prepare_data(
        sequence_length=SEQUENCE_LENGTH,
        test_size=TEST_SIZE
    )
    
    # 2. Create datasets and dataloaders
    train_dataset = MultiStepAQIDataset(data['X_train'], data['y_train'])
    test_dataset = MultiStepAQIDataset(data['X_test'], data['y_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize model
    model = MultiStepAQILSTM(
        input_size=8,
        hidden_size=64,
        num_layers=2,
        forecast_horizon=FORECAST_HORIZON,
        dropout=0.2
    )
    
    print(f"\n🏗️  Model architecture:")
    print(model)
    print(f"\n   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Train model
    trainer = MultiStepAQITrainer(model, device=device)
    trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    # 5. Evaluate model
    results = trainer.evaluate(
        test_loader, 
        data['target_scaler'],
        FORECAST_HORIZON
    )
    
    # 6. Visualize results
    plot_multi_step_results(results, trainer, FORECAST_HORIZON)
    
    # 7. Predict next 8-12 hours
    print(f"\n🔮 Predicting next {FORECAST_HORIZON} hours of AQI...")
    
    # Get last 24 hours of data
    last_sequence = processor.df[processor.feature_columns].values[-SEQUENCE_LENGTH:]
    future_predictions = trainer.predict_future(
        last_sequence,
        data['feature_scaler'],
        data['target_scaler'],
        FORECAST_HORIZON
    )
    
    # Get current timestamp and calculate future times
    last_timestamp = processor.df['date'].iloc[-1]
    current_aqi = processor.df['aqi'].values[-1]
    
    print(f"\n   📅 Current Time: {last_timestamp}")
    print(f"   📊 Current AQI: {current_aqi:.0f}")
    
    # Create forecast table
    print("\n" + "="*80)
    print("📋 12-HOUR AQI FORECAST TABLE")
    print("="*80)
    print(f"{'Hour':<6} {'Date & Time':<22} {'Predicted AQI':<15} {'Category':<25} {'Health Impact'}")
    print("-"*80)
    
    for i, pred_aqi in enumerate(future_predictions, 1):
        # Calculate future timestamp
        future_time = last_timestamp + pd.Timedelta(hours=i)
        time_str = future_time.strftime('%Y-%m-%d %I:%M %p')
        
        # Determine AQI category and health impact
        if pred_aqi <= 50:
            category = "Good 🟢"
            health = "Air quality is satisfactory"
        elif pred_aqi <= 100:
            category = "Moderate 🟡"
            health = "Acceptable for most people"
        elif pred_aqi <= 150:
            category = "Unhealthy (Sensitive) 🟠"
            health = "Sensitive groups affected"
        elif pred_aqi <= 200:
            category = "Unhealthy 🔴"
            health = "Everyone may feel effects"
        elif pred_aqi <= 300:
            category = "Very Unhealthy 🟣"
            health = "Health alert for everyone"
        else:
            category = "Hazardous 🟤"
            health = "Emergency conditions"
        
        print(f"+{i:<5} {time_str:<22} {pred_aqi:>6.1f} AQI      {category:<25} {health}")
    
    print("-"*80)
    print(f"{'SUMMARY':<6} {'':22} {'Avg: ' + f'{future_predictions.mean():.1f}':>15} "
          f"{'Min: ' + f'{future_predictions.min():.1f}':>10} {'Max: ' + f'{future_predictions.max():.1f}':>10}")
    print("="*80)
    
    print("\n" + "="*70)
    print("✅ MULTI-STEP PREDICTION MODEL COMPLETE!")
    print("="*70)
    
    return trainer, processor, results, future_predictions


if __name__ == "__main__":
    trainer, processor, results, future_predictions = main()