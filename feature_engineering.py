import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_financial_ratios(df):
    """
    Create financial ratios and derived features
    """
    print("\nCreating financial ratios and derived features...")
    
    # Make a copy to avoid modifying the original dataframe
    df_featured = df.copy()
    
    # Get numeric columns
    numeric_cols = df_featured.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Create new features only if the required columns exist
    new_features_count = 0
    
    # Common financial columns
    asset_cols = [col for col in numeric_cols if 'asset' in col.lower()]
    liability_cols = [col for col in numeric_cols if 'liabilit' in col.lower()]
    revenue_cols = [col for col in numeric_cols if 'revenue' in col.lower() or 'sales' in col.lower()]
    expense_cols = [col for col in numeric_cols if 'expense' in col.lower() or 'cost' in col.lower()]
    profit_cols = [col for col in numeric_cols if 'profit' in col.lower() or 'income' in col.lower() or 'earnings' in col.lower()]
    
    # 1. Profitability Ratios
    if revenue_cols and profit_cols:
        # Profit Margin
        revenue_col = revenue_cols[0]
        profit_col = profit_cols[0]
        
        # Safe division function to avoid division by zero
        def safe_divide(a, b):
            return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        
        df_featured['profit_margin'] = safe_divide(df_featured[profit_col], df_featured[revenue_col])
        new_features_count += 1
    
    # 2. Liquidity Ratios
    if asset_cols and liability_cols:
        # Current Ratio
        asset_col = asset_cols[0]
        liability_col = liability_cols[0]
        df_featured['current_ratio'] = safe_divide(df_featured[asset_col], df_featured[liability_col])
        new_features_count += 1
    
    # 3. Growth Rates (if time series data)
    date_cols = [col for col in df_featured.columns if 'date' in col.lower()]
    if date_cols and revenue_cols:
        date_col = date_cols[0]
        revenue_col = revenue_cols[0]
        
        # Sort by date
        if pd.api.types.is_datetime64_any_dtype(df_featured[date_col]):
            df_featured = df_featured.sort_values(by=date_col)
            
            # Calculate revenue growth
            df_featured['revenue_growth'] = df_featured[revenue_col].pct_change()
            new_features_count += 1
            
            # Calculate profit growth if profit column exists
            if profit_cols:
                profit_col = profit_cols[0]
                df_featured['profit_growth'] = df_featured[profit_col].pct_change()
                new_features_count += 1
    
    # 4. Expense Ratio
    if expense_cols and revenue_cols:
        expense_col = expense_cols[0]
        revenue_col = revenue_cols[0]
        df_featured['expense_ratio'] = safe_divide(df_featured[expense_col], df_featured[revenue_col])
        new_features_count += 1
    
    # 5. Create lag features for time series analysis
    if date_cols:
        date_col = date_cols[0]
        
        if pd.api.types.is_datetime64_any_dtype(df_featured[date_col]):
            df_featured = df_featured.sort_values(by=date_col)
            
            # Create lag features for important numeric columns
            for col in numeric_cols:
                if col in revenue_cols or col in profit_cols:
                    # Lag 1 period
                    df_featured[f'{col}_lag_1'] = df_featured[col].shift(1)
                    new_features_count += 1
                    
                    # Lag 3 periods
                    df_featured[f'{col}_lag_3'] = df_featured[col].shift(3)
                    new_features_count += 1
                    
                    # Moving average
                    df_featured[f'{col}_ma_3'] = df_featured[col].rolling(window=3, min_periods=1).mean()
                    new_features_count += 1
    
    # 6. Interaction features
    if revenue_cols and expense_cols:
        revenue_col = revenue_cols[0]
        expense_col = expense_cols[0]
        df_featured['revenue_expense_ratio'] = safe_divide(df_featured[revenue_col], df_featured[expense_col])
        new_features_count += 1
    
    # 7. Polynomial features for important columns
    for col in numeric_cols:
        if col in revenue_cols or col in profit_cols:
            # Square
            df_featured[f'{col}_squared'] = np.square(df_featured[col])
            new_features_count += 1
    
    # 8. Logarithmic transformations for skewed data
    for col in numeric_cols:
        # Only apply log to positive columns
        if df_featured[col].min() > 0:
            df_featured[f'{col}_log'] = np.log1p(df_featured[col])
            new_features_count += 1
    
    # Handle infinite values and very large numbers
    # Replace inf with NaN first
    df_featured = df_featured.replace([np.inf, -np.inf], np.nan)
    
    # For numeric columns, fill NaN with median
    numeric_cols = df_featured.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        median_val = df_featured[col].median()
        df_featured[col] = df_featured[col].fillna(median_val)
        
        # Also cap very large values at 99.9th percentile to avoid numerical issues
        upper_limit = df_featured[col].quantile(0.999)
        if not np.isnan(upper_limit):
            df_featured[col] = df_featured[col].clip(upper=upper_limit)
    
    print(f"Created {new_features_count} new features")
    
    return df_featured

def select_features(X, y, k=10):
    """
    Select the most important features using multiple methods
    """
    # Handle any remaining NaN or infinite values
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Ensure y is also clean
    y = y.fillna(y.median())
    
    # 1. Correlation with target
    correlation = X.corrwith(y).abs().sort_values(ascending=False)
    
    # 2. F-regression for linear relationship
    f_selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
    try:
        f_selector.fit(X, y)
        f_scores = pd.Series(f_selector.scores_, index=X.columns)
        f_scores = f_scores.sort_values(ascending=False)
    except Exception as e:
        print(f"Error in f_regression: {e}")
        f_scores = pd.Series(0, index=X.columns)
    
    # 3. Mutual information for non-linear relationship
    mi_selector = SelectKBest(mutual_info_regression, k=min(k, X.shape[1]))
    try:
        mi_selector.fit(X, y)
        mi_scores = pd.Series(mi_selector.scores_, index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
    except Exception as e:
        print(f"Error in mutual_info_regression: {e}")
        mi_scores = pd.Series(0, index=X.columns)
    
    # Combine the methods (simple average of ranks)
    combined_ranks = pd.DataFrame({
        'Correlation': correlation.rank(ascending=False),
        'F-Score': f_scores.rank(ascending=False),
        'MI-Score': mi_scores.rank(ascending=False)
    })
    
    avg_rank = combined_ranks.mean(axis=1).sort_values()
    selected_features = avg_rank.index[:min(k, len(avg_rank))].tolist()
    
    # Visualize feature importance
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    # Plot correlation
    plt.subplot(3, 1, 1)
    correlation.head(10).plot(kind='bar')
    plt.title('Correlation with Target')
    plt.tight_layout()
    
    # Plot F-scores
    plt.subplot(3, 1, 2)
    f_scores.head(10).plot(kind='bar')
    plt.title('F-Regression Scores')
    plt.tight_layout()
    
    # Plot MI-scores
    plt.subplot(3, 1, 3)
    mi_scores.head(10).plot(kind='bar')
    plt.title('Mutual Information Scores')
    plt.tight_layout()
    
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    return selected_features, correlation, f_scores, mi_scores

def apply_pca(X, n_components=0.95):
    """
    Apply PCA for dimensionality reduction
    """
    # Handle any remaining NaN or infinite values
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Standardize the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    try:
        X_pca = pca.fit_transform(X_scaled)
        
        # Visualize explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Components')
        plt.grid(True)
        plt.savefig('plots/pca_explained_variance.png')
        plt.close()
        
        print(f"PCA reduced dimensions from {X.shape[1]} to {X_pca.shape[1]} components")
        print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.2f}")
        
        return X_pca, pca
    except Exception as e:
        print(f"Error in PCA: {e}")
        return X_scaled, None
