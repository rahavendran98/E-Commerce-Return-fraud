"""
Feature engineering functions for creating and transforming features.

This module contains functions for feature creation, selection,
and transformation.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from .utils import logger


def encode_categorical(df: pd.DataFrame, columns: List[str], 
                      method: str = "label") -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical variables.

    Args:
        df: Input DataFrame
        columns: List of categorical columns
        method: Encoding method ('label' or 'onehot')

    Returns:
        Tuple of (encoded DataFrame, encoders dictionary)

    Raises:
        ValueError: If method is not recognized
    """
    df = df.copy()
    encoders = {}
    
    valid_methods = ['label', 'onehot']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
        
        if method == 'label':
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder
            logger.info(f"Label encoded {col}")
        
        elif method == 'onehot':
            # Create dummy variables
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
            encoders[col] = list(dummies.columns)
            logger.info(f"One-hot encoded {col}")
    
    return df, encoders


def scale_numerical(df: pd.DataFrame, columns: List[str],
                   method: str = "standard") -> Tuple[pd.DataFrame, Dict]:
    """
    Scale numerical features.

    Args:
        df: Input DataFrame
        columns: List of numerical columns
        method: Scaling method ('standard', 'minmax', 'robust')

    Returns:
        Tuple of (scaled DataFrame, scalers dictionary)

    Raises:
        ValueError: If method is not recognized
    """
    df = df.copy()
    scalers = {}
    
    valid_methods = ['standard', 'minmax', 'robust']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
        
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        
        df[[col]] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
        logger.info(f"Scaled {col} using {method} scaler")
    
    return df, scalers


def create_interaction_features(df: pd.DataFrame, 
                               feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction features from feature pairs.

    Args:
        df: Input DataFrame
        feature_pairs: List of (col1, col2) tuples to interact

    Returns:
        DataFrame with interaction features added
    """
    df = df.copy()
    
    for col1, col2 in feature_pairs:
        if col1 not in df.columns or col2 not in df.columns:
            logger.warning(f"Columns {col1} or {col2} not found")
            continue
        
        new_col_name = f"{col1}_x_{col2}"
        df[new_col_name] = df[col1] * df[col2]
        logger.info(f"Created interaction feature: {new_col_name}")
    
    return df


def create_polynomial_features(df: pd.DataFrame, columns: List[str],
                              degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial features.

    Args:
        df: Input DataFrame
        columns: List of columns for polynomial expansion
        degree: Polynomial degree

    Returns:
        DataFrame with polynomial features added
    """
    df = df.copy()
    
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    if not columns:
        return df
    
    numeric_cols = [c for c in columns if c in df.columns]
    if not numeric_cols:
        return df
    
    poly_features = poly.fit_transform(df[numeric_cols])
    feature_names = poly.get_feature_names_out(numeric_cols)
    
    # Add new polynomial features
    for i, feat_name in enumerate(feature_names):
        if feat_name not in df.columns:
            df[feat_name] = poly_features[:, i]
    
    logger.info(f"Created {len(feature_names)} polynomial features")
    return df


def select_features(X: pd.DataFrame, y: pd.Series, k: int = 10,
                   method: str = "f_classif") -> Tuple[List[str], Dict]:
    """
    Select most important features using statistical methods.

    Args:
        X: Feature matrix
        y: Target variable
        k: Number of features to select
        method: Selection method ('f_classif' or 'mutual_info')

    Returns:
        Tuple of (selected feature names, scores dictionary)

    Raises:
        ValueError: If method is not recognized
    """
    valid_methods = ['f_classif', 'mutual_info']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")
    
    score_func = f_classif if method == 'f_classif' else mutual_info_classif
    selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
    
    selector.fit(X, y)
    
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    selected_features = feature_scores.head(k)['feature'].tolist()
    scores_dict = dict(zip(feature_scores['feature'], feature_scores['score']))
    
    logger.info(f"Selected {len(selected_features)} features using {method}")
    
    return selected_features, scores_dict


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names

    Returns:
        DataFrame with features sorted by importance
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df


def get_feature_correlation(df: pd.DataFrame, target_col: Optional[str] = None,
                           threshold: float = 0.7) -> pd.DataFrame:
    """
    Get correlation matrix and identify highly correlated features.

    Args:
        df: Input DataFrame
        target_col: Target column name (optional)
        threshold: Correlation threshold for reporting

    Returns:
        DataFrame of high correlations
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        corr_df = pd.DataFrame(high_corr_pairs).sort_values(
            'correlation', ascending=False, key=abs
        )
        logger.info(f"Found {len(corr_df)} feature pairs with |correlation| > {threshold}")
        return corr_df
    
    logger.info(f"No features with |correlation| > {threshold}")
    return pd.DataFrame()


def handle_imbalanced_data(X: pd.DataFrame, y: pd.Series,
                          method: str = "undersample") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using resampling.

    Args:
        X: Feature matrix
        y: Target variable
        method: Resampling method ('oversample', 'undersample', 'smote')

    Returns:
        Tuple of (resampled features, resampled target)

    Raises:
        ValueError: If method is not recognized
    """
    valid_methods = ['oversample', 'undersample', 'smote']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")
    
    if method == 'oversample':
        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(random_state=42)
    elif method == 'undersample':
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=42)
    elif method == 'smote':
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=42)
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    logger.info(f"Applied {method} resampling")
    logger.info(f"Original distribution: {y.value_counts().to_dict()}")
    logger.info(f"Resampled distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
    
    return X_resampled, y_resampled


def create_feature_engineering_pipeline(df: pd.DataFrame, target_col: str,
                                       config: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply complete feature engineering pipeline.

    Args:
        df: Input DataFrame
        target_col: Target column name
        config: Configuration dictionary with feature engineering parameters

    Returns:
        Tuple of (engineered DataFrame, list of final features)
    """
    df = df.copy()
    logger.info("Starting feature engineering pipeline")
    
    # Create interaction features
    if config.get('interaction_features'):
        df = create_interaction_features(df, config['interaction_features'])
    
    # Create polynomial features
    if config.get('polynomial_features'):
        df = create_polynomial_features(df, config['polynomial_features'])
    
    # Encode categorical variables
    categorical_cols = config.get('categorical_columns', [])
    if categorical_cols:
        df, _ = encode_categorical(df, categorical_cols, method='label')
    
    # Scale numerical features
    numerical_cols = config.get('numerical_columns', [])
    if numerical_cols:
        df, _ = scale_numerical(df, numerical_cols, method='standard')
    
    # Remove target column from features
    final_features = [c for c in df.columns if c != target_col]
    
    logger.info(f"Feature engineering complete. {len(final_features)} features")
    return df, final_features
