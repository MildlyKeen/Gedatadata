import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
from data_preparation import load_and_clean_data, prepare_features
from feature_engineering import create_financial_ratios, select_features

def train_test_data_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """
    Train multiple regression models
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    trained_models = {}
    cv_scores = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Cross-validation
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_scores[name] = cv_score
        print(f"{name} CV R² Score: {cv_score.mean():.4f} ± {cv_score.std():.4f}")
    
    return trained_models, cv_scores

def evaluate_models(models, X_test, y_test):
    """
    Evaluate models on test data
    """
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{name} - Actual vs Predicted')
        plt.tight_layout()
        plt.savefig(f'plots/{name.replace(" ", "_")}_actual_vs_predicted.png')
        plt.close()
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Plot model comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='R²', data=results_df)
    plt.title('Model Comparison - R² Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/model_comparison_r2.png')
    plt.close()
    
    return results_df

def tune_best_model(X_train, y_train, X_test, y_test, best_model_name):
    """
    Perform hyperparameter tuning on the best model
    """
    if best_model_name == 'Random Forest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    elif best_model_name == 'Ridge Regression':
        model = Ridge(random_state=42)
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr']
        }
    elif best_model_name == 'Lasso Regression':
        model = Lasso(random_state=42)
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'selection': ['cyclic', 'random']
        }
    else:  # Linear Regression
        print("Linear Regression doesn't have hyperparameters to tune")
        model = LinearRegression()
        return model.fit(X_train, y_train)
    
    print(f"\nTuning hyperparameters for {best_model_name}...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV R² Score: {grid_search.best_score_:.4f}")
    
    # Evaluate the tuned model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Tuned model test R² Score: {r2:.4f}")
    
    return best_model

def feature_importance(model, feature_names, model_name):
    """
    Extract and plot feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'plots/{model_name.replace(" ", "_")}_feature_importance.png')
        plt.close()
        
        # Return feature importance as dataframe
        return pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
    elif hasattr(model, 'coef_'):
        coefs = model.coef_
        abs_coefs = np.abs(coefs)
        indices = np.argsort(abs_coefs)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Coefficients - {model_name}')
        plt.bar(range(len(indices)), coefs[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'plots/{model_name.replace(" ", "_")}_coefficients.png')
        plt.close()
        
        # Return coefficients as dataframe
        return pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Coefficient': coefs[indices]
        })
    else:
        print(f"Model {model_name} doesn't have feature importances or coefficients")
        return None

if __name__ == "__main__":
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load the featured data
    try:
        df_featured = pd.read_csv('featured_financials.csv')
        print("Loaded featured data from file")
    except:
        print("Featured data file not found, loading and preparing data")
        df_cleaned = load_and_clean_data()
        df_featured = create_financial_ratios(df_cleaned)
    
    # Assume 'profit' or 'revenue' as target variable (adjust based on your data)
    target_column = 'profit' if 'profit' in df_featured.columns else 'revenue'
    if target_column in df_featured.columns:
        # Prepare features
        X = df_featured.drop(columns=[target_column])
        y = df_featured[target_column]
        
        # Handle non-numeric columns
        X = X.select_dtypes(include=['float64', 'int64'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_data_split(X, y)
        
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
        importance_df = feature_importance(tuned_model, X.columns, best_model_name)
        if importance_df is not None:
            print("\nFeature Importance/Coefficients:")
            print(importance_df.head(10))
        
        # Save the best model
        joblib.dump(tuned_model, f'models/best_model_{best_model_name.replace(" ", "_")}.pkl')
        print(f"\nBest model saved to 'models/best_model_{best_model_name.replace(' ', '_')}.pkl'")
    else:
        print(f"Target column '{target_column}' not found in the dataset")
