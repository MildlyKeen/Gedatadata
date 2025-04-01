import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_clean_data(file_path='Gedatadata Financials.csv'):
    """
    Load and clean the financial dataset
    """
    print(f"Loading data from {file_path}...")
    
    # Load the data
    df = pd.read_csv(file_path, on_bad_lines="skip", encoding="utf-8")
    
    # Display basic information
    print("\nOriginal data shape:", df.shape)
    print("\nFirst few rows of original data:")
    print(df.head())
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # Convert financial columns to numeric
    def convert_to_float(s):
        if isinstance(s, str):
            s = s.strip().replace("$", "").replace(",", "").replace("(", "-").replace(")", "")
            if s == "-":
                return float(0)
            else:
                try:
                    return float(s)
                except ValueError:
                    return np.nan
        return s
    
    # Identify money columns
    money_cols = df.columns[df.columns.str.contains('price|cost|revenue|profit|margin|amount|value|sales', case=False)]
    
    # Apply conversion to all money columns
    for col in money_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_to_float)
    
    # Handle date formatting
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for date_col in date_columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        # Fill missing values with median for numeric columns
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    # Fill categorical columns with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isna().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    
    print("\nCleaned data shape:", df_cleaned.shape)
    print("\nMissing values after cleaning:")
    print(df_cleaned.isna().sum().sum())
    
    return df_cleaned

def explore_data(df):
    """
    Perform exploratory data analysis on the cleaned dataset
    """
    print("\n=== Exploratory Data Analysis ===")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Distribution of numeric features
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols[:5]:  # Limit to first 5 numeric columns for brevity
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'plots/{col}_distribution.png')
        plt.close()
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    return corr_matrix

def prepare_features(df, target_column=None):
    """
    Prepare features for machine learning
    """
    # If target column is specified, separate features and target
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        # If no target is specified, assume all numeric columns are features
        X = df.select_dtypes(include=['float64', 'int64'])
        y = None
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = X_processed.select_dtypes(include=['float64', 'int64']).columns
    X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
    
    print(f"\nPrepared {X_processed.shape[1]} features for machine learning")
    
    return X_processed, y, scaler

if __name__ == "__main__":
    # Execute the data preparation pipeline
    df_cleaned = load_and_clean_data()
    corr_matrix = explore_data(df_cleaned)
    
    # Save the cleaned data
    df_cleaned.to_csv('cleaned_financials.csv', index=False)
    print("\nCleaned data saved to 'cleaned_financials.csv'")
