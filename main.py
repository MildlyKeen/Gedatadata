import os
import argparse
import pandas as pd
import time

def run_pipeline(input_file, target_column=None, test_size=0.2, forecast_periods=12):
    """
    Run the complete machine learning pipeline
    """
    start_time = time.time()
    
    print("="*80)
    print("FINANCIAL DATA MACHINE LEARNING PIPELINE")
    print("="*80)
    
    # Step 1: Data Preparation
    print("\n[STEP 1] Data Preparation and Cleaning")
    from data_preparation import load_and_clean_data, explore_data
    
    df_cleaned = load_and_clean_data(input_file)
    corr_matrix = explore_data(df_cleaned)
    df_cleaned.to_csv('cleaned_financials.csv', index=False)
    print("Data preparation completed. Cleaned data saved to 'cleaned_financials.csv'")
    
    # Step 2: Feature Engineering
    print("\n[STEP 2] Feature Engineering and Selection")
    from feature_engineering import create_financial_ratios, select_features, apply_pca
    
    df_featured = create_financial_ratios(df_cleaned)
    
    # Determine target column if not specified
    if target_column is None:
        for col in ['profit', 'revenue', 'income', 'earnings']:
            if col in df_featured.columns:
                target_column = col
                break
        if target_column is None:
            # If still not found, use the last numeric column
            numeric_cols = df_featured.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                target_column = numeric_cols[-1]
            else:
                raise ValueError("No suitable target column found. Please specify a target column.")
    
    print(f"Using '{target_column}' as the target variable")
    
    if target_column in df_featured.columns:
        X = df_featured.drop(columns=[target_column])
        y = df_featured[target_column]
        
        # Handle non-numeric columns
        X_numeric = X.select_dtypes(include=['float64', 'int64'])
        
        # Select important features
        selected_features, correlation, f_scores, mi_scores = select_features(X_numeric, y)
        
        # Apply PCA
        X_selected = X_numeric[selected_features]
        X_pca, pca = apply_pca(X_selected)
        
        # Save the featured data
        df_featured.to_csv('featured_financials.csv', index=False)
        print(f"Feature engineering completed. Featured data saved to 'featured_financials.csv'")
    else:
        raise ValueError(f"Target column '{target_column}' not found in the dataset")
    
    # Step 3: Model Development
    print("\n[STEP 3] Model Development and Evaluation")
    from model_development import (
        train_test_data_split, train_models, evaluate_models, 
        tune_best_model, feature_importance
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_data_split(X_numeric, y, test_size=test_size)
    
    # Train models
    trained_models, cv_scores = train_models(X_train, y_train)
    
    # Evaluate models
    results_df = evaluate_models(trained_models, X_test, y_test)
    print("\nModel Evaluation Results:")
    print(results_df)
    
    # Find the best model
    best_model_name = results_df.loc[results_df['R²'].idxmax(), 'Model']
    print(f"\nBest model: {best_model_name} with R² = {results_df['R²'].max():.4f}")
    
    # Tune the best model
    tuned_model = tune_best_model(X_train, y_train, X_test, y_test, best_model_name)
    
    # Get feature importance
    importance_df = feature_importance(tuned_model, X_numeric.columns, best_model_name)
    if importance_df is not None:
        print("\nFeature Importance/Coefficients:")
        print(importance_df.head(10))
        importance_df.to_csv('feature_importance.csv', index=False)
    
    # Save the best model
    import joblib
    os.makedirs('models', exist_ok=True)
    model_path = f'models/best_model_{best_model_name.replace(" ", "_")}.pkl'
    joblib.dump(tuned_model, model_path)
    print(f"\nBest model saved to '{model_path}'")
    
    # Step 4: Model Deployment and Prediction
    print("\n[STEP 4] Model Deployment and Prediction")
    from model_deployment import (
        make_predictions, create_prediction_report, visualize_predictions,
        predict_future, create_forecast_visualization
    )
    
    # Make predictions on the entire dataset
    predictions = make_predictions(tuned_model, X_numeric)
    
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
        future_df = predict_future(tuned_model, last_data_point, X_numeric.columns, 
                                  periods=forecast_periods, date_col=date_col)
        
        # Visualize forecast
        create_forecast_visualization(df_featured, future_df, target_column, date_col)
        
        # Save forecast
        future_df.to_csv('forecast_results.csv', index=False)
        print(f"Forecast results for {forecast_periods} periods saved to 'forecast_results.csv'")
    
    # Calculate execution time
    execution_time = time.time() - start_time
    print(f"\nPipeline completed in {execution_time:.2f} seconds")
    
    # Generate summary report
    generate_summary_report(
        input_file, df_cleaned.shape, df_featured.shape, 
        target_column, best_model_name, results_df, 
        importance_df, execution_time
    )

def generate_summary_report(input_file, cleaned_shape, featured_shape, 
                           target_column, best_model, results_df, 
                           importance_df, execution_time):
    """
    Generate a summary report of the ML pipeline
    """
    with open('ml_pipeline_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("FINANCIAL DATA MACHINE LEARNING PIPELINE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Input File: {input_file}\n")
        f.write(f"Execution Time: {execution_time:.2f} seconds\n\n")
        
        f.write("DATA SUMMARY:\n")
        f.write(f"Original Data Shape: {cleaned_shape}\n")
        f.write(f"Featured Data Shape: {featured_shape}\n")
        f.write(f"Target Variable: {target_column}\n\n")
        
        f.write("MODEL PERFORMANCE:\n")
        f.write(f"Best Model: {best_model}\n")
        f.write(f"Performance Metrics:\n")
        best_model_results = results_df[results_df['Model'] == best_model].to_string(index=False)
        f.write(best_model_results + "\n\n")
        
        f.write("TOP FEATURES:\n")
        if importance_df is not None:
            top_features = importance_df.head(10).to_string(index=False)
            f.write(top_features + "\n\n")
        
        f.write("OUTPUT FILES:\n")
        f.write("- cleaned_financials.csv: Cleaned dataset\n")
        f.write("- featured_financials.csv: Dataset with engineered features\n")
        f.write("- prediction_results.csv: Predictions on the entire dataset\n")
        f.write("- feature_importance.csv: Feature importance/coefficients\n")
        f.write(f"- models/best_model_{best_model.replace(' ', '_')}.pkl: Trained model\n")
        
        if os.path.exists('forecast_results.csv'):
            f.write("- forecast_results.csv: Future predictions\n")
        
        f.write("\nPLOTS:\n")
        for plot in os.listdir('plots'):
            f.write(f"- plots/{plot}\n")
    
    print(f"\nSummary report generated: ml_pipeline_summary.txt")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Financial Data Machine Learning Pipeline')
    parser.add_argument('--input', type=str, default='Gedatadata Financials.csv',
                        help='Input CSV file path')
    parser.add_argument('--target', type=str, default=None,
                        help='Target column name')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size (proportion)')
    parser.add_argument('--forecast', type=int, default=12,
                        help='Number of periods to forecast')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run the pipeline
    run_pipeline(
        input_file=args.input,
        target_column=args.target,
        test_size=args.test_size,
        forecast_periods=args.forecast
    )
