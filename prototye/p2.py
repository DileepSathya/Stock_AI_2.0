import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

class StockRangePredictor:
    def __init__(self, data_path):
        self.df = self.load_and_preprocess(data_path)
        self.features = None
        self.models = {}
        
    @staticmethod
    def load_and_preprocess(data_path):
        """Load and basic preprocessing"""
        df = pd.read_json(data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    def add_cyclic_features(self, col, period):
        """Add cyclic encoding for temporal features"""
        self.df[f'{col}_sin'] = np.sin(2 * np.pi * self.df[col]/period)
        self.df[f'{col}_cos'] = np.cos(2 * np.pi * self.df[col]/period)
        return self.df.drop(col, axis=1)
    
    def calculate_atr(self, window=14):
        """Calculate Average True Range"""
        hl = self.df['high'] - self.df['low']
        hc = abs(self.df['high'] - self.df['close'].shift())
        lc = abs(self.df['low'] - self.df['close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(window).mean()
    
    def calculate_historical_k(self):
        """Calculate historical k-values with outlier handling"""
        self.df['k'] = (self.df['high'] - self.df['open']) / (self.df['high'] - self.df['low'])
        self.df['k'] = self.df['k'].clip(0.2, 0.8)  # Constrain to reasonable range
        self.df['rolling_k'] = self.df.groupby('symbol')['k'].transform(lambda x: x.rolling(30).mean())
        return self.df
    
    def generate_features(self):
        """Generate all features for modeling"""
        # Time features
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['month'] = self.df['date'].dt.month
        self.df = self.add_cyclic_features('day_of_week', 7)
        self.df = self.add_cyclic_features('month', 12)
        
        # Lagged price features
        for col in ['open', 'high', 'low', 'close', 'volume']:
            self.df[f'prev_day_{col}'] = self.df.groupby('symbol')[col].shift(1)
        
        # Range features
        self.df['daily_range'] = self.df['high'] - self.df['low']
        self.df['prev_day_range'] = self.df.groupby('symbol')['daily_range'].shift(1)
        
        # Volatility features
        self.df['atr_14'] = self.calculate_atr()
        self.df['gap_pct'] = (self.df['open'] - self.df['prev_day_close']) / self.df['prev_day_close'] * 100
        
        # EMA features
        for span in [5, 20, 50]:
            self.df[f'ema_{span}'] = self.df.groupby('symbol')['close'].transform(
                lambda x: x.ewm(span=span, adjust=False).mean()
            )
            self.df[f'open_ema_{span}_dist'] = (self.df['open'] - self.df[f'ema_{span}']) / self.df[f'ema_{span}']
        
        # k-value calculation
        self.df = self.calculate_historical_k()
        
        # Target engineering
        self.df['target_range'] = self.df['daily_range'] / self.df['atr_14']
        
        # Define feature set
        self.features = [
            'prev_day_range', 'atr_14', 'gap_pct',
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'open_ema_5_dist', 'open_ema_20_dist', 'open_ema_50_dist',
            'rolling_k'
        ]
        
        return self.df.dropna()
    
    def train_models(self):
        """Train symbol-specific models with time-series validation"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        for symbol in self.df['symbol'].unique():
            symbol_data = self.df[self.df['symbol'] == symbol]
            
            if len(symbol_data) < 100:  # Minimum data requirement
                continue
                
            X = symbol_data[self.features]
            y = symbol_data['target_range']
            
            model = LinearRegression()
            scores = []
            
            # Time-series cross-validation
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                scores.append(mean_absolute_error(y_test, preds))
            
            # Final model training
            model.fit(X, y)
            self.models[symbol] = {
                'model': model,
                'avg_mae': np.mean(scores),
                'last_data': X.iloc[-1:].values
            }
    
    def predict_next_day(self):
        """Generate predictions for all symbols"""
        predictions = {}
        
        for symbol, model_data in self.models.items():
            current_data = self.df[self.df['symbol'] == symbol].iloc[-1]
            
            # Predict normalized range
            pred_range_norm = model_data['model'].predict(model_data['last_data'])[0]
            
            # Convert to absolute range
            pred_range = pred_range_norm * current_data['atr_14']
            
            # Get predicted k (use rolling average or could predict separately)
            pred_k = current_data['rolling_k']
            
            predictions[symbol] = {
                'predicted_range': pred_range,
                'predicted_k': pred_k,
                'current_atr': current_data['atr_14'],
                'model_mae': model_data['avg_mae']
            }
        
        return predictions
    
    def generate_signals(self, current_prices):
        """Convert predictions to trading signals"""
        predictions = self.predict_next_day()
        signals = {}
        
        for symbol, pred in predictions.items():
            if symbol not in current_prices:
                continue
                
            open_price = current_prices[symbol]['open']
            pred_range = pred['predicted_range']
            k = pred['predicted_k']
            
            # Calculate predicted high/low
            high = open_price + (k * pred_range)
            low = open_price - ((1 - k) * pred_range)
            
            signals[symbol] = {
                'symbol': symbol,
                'predicted_high': round(high, 2),
                'predicted_low': round(low, 2),
                'predicted_range': round(pred_range, 2),
                'confidence': min(1.0, pred_range / pred['current_atr']),
                'model_error': round(pred['model_mae'], 4),
                'current_open': open_price
            }
        
        return pd.DataFrame.from_dict(signals, orient='index')

# Example Usage
if __name__ == "__main__":
    # Initialize and process data
    predictor = StockRangePredictor("C:/Users/Dileep Sathya/OneDrive/Desktop/Stock_AI_2.0/artifacts/hist_data.json")
    predictor.generate_features()
    
