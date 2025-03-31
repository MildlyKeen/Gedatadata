import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from data_preparation import load_and_clean_data, prepare_features
from feature_engineering import create_financial_ratios

def load_model(model_path):
    """
    Load the trained model
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def prepare_input_data(input_data, feature_names):
    """
    Prepare input data for prediction
    """
    # Ensure input data has all required features
    for feature in feature_names:
        if feature not in input_data.columns:
            print(f"Warning: Feature '{feature}' not found in input data. Setting to 0.")
            input_data[feature] = 0
    
    # Select only the features used by the model
    input_features = input_data[feature_names]
    
    return input_features

def make_predictions(model, input_data):
    """
    Make predictions using the trained model
    """
    predictions = model.predict(input_data)
    return predictions

def create_prediction_report(input_data, predictions, target_name):
    """
    Create a report with predictions
    """
    # Add predictions to input data
    result_df = input_data.copy()
    result_df[f'Predicted_{target_name}'] = predictions
    
    # If actual values are available, calculate error
    if target_name in result_df.columns:
        result_df[f'Error_{target_name}'] = result_df[target_name] - result_df[f'Predicted_{target_name}']
        result_df[f'Absolute_Error_{target_name}'] = abs(result_df[f'Error_{target_name}'])
        result_df[f'Percentage_Error_{target_name}'] = (
            result_df[f'Error_{target_name}'] / result_df[target_name] * 100
        )
    
    return result_df

def visualize_predictions(result_df, target_name):
    """
    Visualize predictions vs actual values
    """
    if target_name in result_df.columns:
        plt.figure(figsize=(12, 8))
        plt.scatter(result_df[target_name], result_df[f'Predicted_{target_name}'], alpha=0.5)
        plt.plot(
            [result_df[target_name].min(), result_df[target_name].max()],
            [result_df[target_name].min(), result_df[target_name].max()],
            'r--'
        )
        plt.xlabel(f'Actual {target_name}')
        plt.ylabel(f'Predicted {target_name}')
        plt.title(f'Actual vs Predicted {target_name}')
        plt.tight_layout()
        plt.savefig(f'plots/prediction_results_{target_name}.png')
        plt.close()
        
        # Plot error distribution
        plt.figure(figsize=(12, 8))
        plt.hist(result_df[f'Error_{target_name}'], bins=30)
        plt.xlabel(f'Prediction Error ({target_name})')
        plt.ylabel('Frequency')
        plt.title(f'Prediction Error Distribution - {target_name}')
        plt.tight_layout()
        plt.savefig(f'plots/prediction_error_distribution_{target_name}.png')
        plt.close()

def predict_future(model, last_data_point, feature_names, periods=12, date_col=None):
    """
    Generate future predictions based on the last data point
    For time series financial data
    """
    future_predictions = []
    current_data = last_data_point.copy()
    
    for i in range(periods):
        # Prepare the current data point
        input_features = prepare_input_data(pd.DataFrame([current_data]), feature_names)
        
        # Make prediction
        prediction = make_predictions(model, input_features)[0]
        
        # Create a record with the prediction
        future_record = current_data.copy()
        
        # Update date if date column is provided
        if date_col and date_col in future_record:
            if isinstance(future_record[date_col], pd.Timestamp):
                future_record[date_col] = future_record[date_col] + pd.DateOffset(months=1)
            else:
                # Try to parse as date if it's not already a timestamp
                try:
                    date_val = pd.to_datetime(future_record[date_col])
                    future_record[date_col] = date_val + pd.DateOffset(months=1)
                except:
                    future_record[date_col] = f"Period {i+1}"
        
        # Add prediction to the record
        future_record['prediction'] = prediction
        future_predictions.append(future_record)
        
        # Update current data for next iteration (for autoregressive features)
        # This assumes the target is used as a feature in the next prediction
        target_name = 'prediction'  # The name you want to use for the target in future predictions
        
        # Update lag features if they exist
        for feature in feature_names:
            if '_lag_1' in feature and feature.replace('_lag_1', '') in current_data:
                base_feature = feature.replace('_lag_1', '')
                current_data[feature] = current_data[base_feature]
            
            if '_lag_3' in feature and feature.replace('_lag_3', '') in current_data:
                # For lag_3, we would need to keep track of more history
                # This is a simplified approach
                base_feature = feature.replace('_lag_3', '')
                current_data[feature] = current_data[base_feature]
        
        # Update the target value with the prediction for the next iteration
        if target_name in feature_names:
            current_data[target_name] = prediction
    
    # Create a dataframe with all future predictions
    future_df = pd.DataFrame(future_predictions)
    
    return future_df

def create_forecast_visualization(historical_data, future_data, target_column, date_column=None):
    """
    Create visualization of historical data and forecast
    """
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    if date_column and date_column in historical_data.columns:
        plt.plot(
            historical_data[date_column], 
            historical_data[target_column], 
            'b-', 
            label='Historical'
        )
        
        # Plot future predictions
        if date_column in future_data.columns:
            plt.plot(
                future_data[date_column], 
                future_data['prediction'], 
                'r--', 
                label='Forecast'
            )
        else:
            # If no date column in future data, use indices
            plt.plot(
                range(
                    len(historical_data), 
                    len(historical_data) + len(future_data)
                ), 
                future_data['prediction'], 
                'r--', 
                label='Forecast'
            )
    else:
        # If no date column, use indices
        plt.plot(
            range(len(historical_data)), 
            historical_data[target_column], 
            'b-', 
            label='Historical'
        )
        plt.plot(
            range(
                len(historical_data), 
                len(historical_data) + len(future_data)
            ), 
            future_data['prediction'], 
            'r--', 
            label='Forecast'
        )
    
    plt.xlabel('Time')
    plt.ylabel(target_column)
    plt.title(f'{target_column} - Historical Data and Forecast')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/forecast_{target_column}.png')
    plt.close()

if __name__ == "__main__":
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Find the best model
    model_dir = 'models'
    model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_')]
    
    if not model_files:
        print("No trained models found. Please run model_development.py first.")
    else:
        # Load the best model
        best_model_path = os.path.join(model_dir, model_files[0])
        model = load_model(best_model_path)
        
        if model:
            # Load test data for prediction
            try:
                df_featured = pd.read_csv('featured_financials.csv')
                print("Loaded featured data from file")
            except:
                print("Featured data file not found, loading and preparing data")
                df_cleaned = load_and_clean_data()
                df_featured = create_financial_ratios(df_cleaned)
            
            # Determine target column
            target_column = 'profit' if 'profit' in df_featured.columns else 'revenue'
            
            # Prepare features
            X = df_featured.drop(columns=[target_column])
            y = df_featured[target_column]
            
            # Handle non-numeric columns
            X = X.select_dtypes(include=['float64', 'int64'])
            feature_names = X.columns.tolist()
            
            # Make predictions on the entire dataset
            predictions = make_predictions(model, X)
            
            # Create prediction report
            result_df = create_prediction_report(df_featured, predictions, target_column)
            
            # Visualize predictions
            visualize_predictions(result_df, target_column)
            
            # Save prediction results
            result_df.to_csv('prediction_results.csv', index=False)
            print("Prediction results saved to 'prediction_results.csv'")
            
            # Generate future predictions (if time series data)
            date_columns = [col for col in df_featured.columns if 'date' in col.lower()]
            date_col = date_columns[0] if date_columns else None
            
            if date_col:
                print(f"\nGenerating future predictions based on date column: {date_col}")
                # Sort by date
                df_featured = df_featured.sort_values(by=date_col)
                
                # Get the last data point
                last_data_point = df_featured.iloc[-1].to_dict()
                
                # Predict future periods
                future_df = predict_future(model, last_data_point, feature_names, periods=12, date_col=date_col)
                
                # Visualize forecast
                create_forecast_visualization(df_featured, future_df, target_column, date_col)
                
                # Save forecast
                future_df.to_csv('forecast_results.csv', index=False)
                print("Forecast results saved to 'forecast_results.csv'")
