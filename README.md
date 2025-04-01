# Gedatadata
# Gedatadata# Financial Data Machine Learning Project

This project implements a complete machine learning pipeline for financial data analysis. It includes data cleaning, feature engineering, model development, and forecasting capabilities.

## Project Structure

- `data_preparation.py`: Data loading, cleaning, and exploratory analysis
- `feature_engineering.py`: Feature creation, selection, and dimensionality reduction
- `model_development.py`: Model training, evaluation, and hyperparameter tuning
- `model_deployment.py`: Making predictions and generating forecasts
- `main.py`: Orchestrates the entire pipeline

## Requirements

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage

### Running the Complete Pipeline

To run the complete pipeline with default settings:

```bash
python main.py
```

### Command Line Arguments

- `--input`: Input CSV file path (default: 'Gedatadata Financials.csv')
- `--target`: Target column name (default: auto-detected)
- `--test_size`: Test set size proportion (default: 0.2)
- `--forecast`: Number of periods to forecast (default: 12)


python main.py --input Financials.csv --target profit --test_size 0.25 --forecast 24


## Pipeline Steps

1. **Data Preparation**
   - Load and clean the financial data
   - Handle missing values and outliers
   - Perform exploratory data analysis

2. **Feature Engineering**
   - Create financial ratios and derived features
   - Select the most important features
   - Apply dimensionality reduction (PCA)

3. **Model Development**
   - Train multiple regression models
   - Evaluate model performance
   - Tune hyperparameters of the best model

4. **Model Deployment**
   - Make predictions on the entire dataset
   - Generate future forecasts (if time series data)
   - Visualize results

## Output Files

- `cleaned_financials.csv`: Cleaned dataset
- `featured_financials.csv`: Dataset with engineered features
- `prediction_results.csv`: Predictions on the entire dataset
- `feature_importance.csv`: Feature importance/coefficients
- `forecast_results.csv`: Future predictions (if applicable)
- `ml_pipeline_summary.txt`: Summary of the pipeline execution
- `models/`: Directory containing the trained model
- `plots/`: Directory containing visualizations

## Visualizations

The pipeline generates various visualizations:
- Distribution of features
- Correlation matrix
- Feature importance
- Actual vs. predicted values
- Prediction error distribution
- Forecast visualization (if time series data)

## Customization

You can customize the pipeline by modifying the individual Python files:
- Add new feature engineering techniques in `feature_engineering.py`
- Implement additional models in `model_development.py`
- Enhance visualization in `model_deployment.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

## Summary

This complete machine learning project for financial data analysis includes:

1. **Data Preparation**: Cleaning, handling missing values, and exploratory analysis
2. **Feature Engineering**: Creating financial ratios, feature selection, and dimensionality reduction
3. **Model Development**: Training multiple models, evaluation, and hyperparameter tuning
4. **Model Deployment**: Making predictions and generating forecasts
5. **Pipeline Orchestration**: A main script to run the entire process
6. **Documentation**: A comprehensive README with usage instructions

The project is designed to be flexible and can work with various financial datasets. It automatically detects appropriate target variables and handles both cross-sectional and time series financial data.

To run the project, simply execute:

```bash
python main.py --input Financials.csv --target profit --test_size 0.25 --forecast 24

