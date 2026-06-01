"""
Model utilities and definitions for training and evaluation.

This module contains functions for model training, evaluation,
and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from .utils import logger
from .config import RANDOM_SEED


def train_test_validation_split(X: pd.DataFrame, y: pd.Series,
                               train_size: float = 0.7,
                               val_size: float = 0.15,
                               test_size: float = 0.15,
                               random_state: int = RANDOM_SEED
                               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                         pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Feature matrix
        y: Target variable
        train_size: Training set proportion
        val_size: Validation set proportion
        test_size: Test set proportion
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=train_size,
        random_state=random_state,
        stratify=y
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_ratio,
        random_state=random_state,
        stratify=y_temp
    )
    
    logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    logger.info(f"Train distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Val distribution: {y_val.value_counts().to_dict()}")
    logger.info(f"Test distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_model(model_name: str, params: Optional[Dict] = None) -> Any:
    """
    Get a model instance by name.

    Args:
        model_name: Name of the model ('logistic_regression', 'random_forest', 'xgboost')
        params: Model parameters (optional)

    Returns:
        Initialized model instance

    Raises:
        ValueError: If model name is not recognized
    """
    models = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
    }
    
    if model_name not in models:
        # Try importing xgboost
        if model_name == 'xgboost':
            try:
                from xgboost import XGBClassifier
                models['xgboost'] = XGBClassifier
            except ImportError:
                raise ValueError(f"Model '{model_name}' requires xgboost library")
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    model_class = models[model_name]
    
    if params:
        model = model_class(**params)
        logger.info(f"Created {model_name} with custom parameters")
    else:
        model = model_class(random_state=RANDOM_SEED)
        logger.info(f"Created {model_name} with default parameters")
    
    return model


def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Train a model.

    Args:
        model: Model instance
        X_train: Training features
        y_train: Training target

    Returns:
        Trained model
    """
    model.fit(X_train, y_train)
    logger.info(f"Model trained on {len(X_train)} samples")
    return model


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series,
                  dataset_name: str = "Test") -> Dict[str, float]:
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X: Features
        y: Target
        dataset_name: Name of dataset for logging

    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba),
    }
    
    logger.info(f"\n{dataset_name} Set Metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def get_detailed_metrics(model: Any, X: pd.DataFrame, y: pd.Series,
                        class_names: Optional[Dict[int, str]] = None
                        ) -> Dict[str, Any]:
    """
    Get detailed evaluation metrics including confusion matrix and classification report.

    Args:
        model: Trained model
        X: Features
        y: Target
        class_names: Dictionary mapping class indices to names

    Returns:
        Dictionary with detailed metrics
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Basic metrics
    basic_metrics = evaluate_model(model, X, y)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Classification report
    if class_names:
        target_names = [class_names.get(i, str(i)) for i in sorted(set(y))]
    else:
        target_names = [str(i) for i in sorted(set(y))]
    
    class_report = classification_report(y, y_pred, target_names=target_names, output_dict=True)
    
    # ROC curve data
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    
    detailed_metrics = {
        **basic_metrics,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
    }
    
    return detailed_metrics


def hyperparameter_tuning(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                         param_grid: Dict[str, list],
                         cv: int = 5) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform hyperparameter tuning using GridSearchCV.

    Args:
        model: Model instance
        X_train: Training features
        y_train: Training target
        param_grid: Parameter grid for tuning
        cv: Number of cross-validation folds

    Returns:
        Tuple of (best model, best parameters)
    """
    logger.info(f"Starting hyperparameter tuning with {cv}-fold CV")
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best CV score: {best_score:.4f}")
    
    return best_model, best_params


def compare_models(models_dict: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Train and compare multiple models.

    Args:
        models_dict: Dictionary of {model_name: model_instance}
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, model in models_dict.items():
        logger.info(f"\nTraining {model_name}...")
        
        # Train
        trained_model = train_model(model, X_train, y_train)
        
        # Evaluate
        train_metrics = evaluate_model(trained_model, X_train, y_train, "Train")
        test_metrics = evaluate_model(trained_model, X_test, y_test, "Test")
        
        # Combine results
        results.append({
            'model': model_name,
            'train_accuracy': train_metrics['accuracy'],
            'test_accuracy': test_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'test_precision': test_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'test_recall': test_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'test_f1': test_metrics['f1'],
            'train_roc_auc': train_metrics['roc_auc'],
            'test_roc_auc': test_metrics['roc_auc'],
        })
    
    comparison_df = pd.DataFrame(results)
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    
    return comparison_df


def get_cross_validation_scores(model: Any, X: pd.DataFrame, y: pd.Series,
                               cv: int = 5, scoring: str = 'roc_auc') -> Dict[str, float]:
    """
    Get cross-validation scores.

    Args:
        model: Model instance
        X: Features
        y: Target
        cv: Number of cross-validation folds
        scoring: Scoring metric

    Returns:
        Dictionary with CV scores and statistics
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    cv_metrics = {
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'scores': scores.tolist(),
    }
    
    logger.info(f"{cv}-Fold Cross-Validation ({scoring}):")
    logger.info(f"  Mean: {cv_metrics['mean']:.4f} (+/- {cv_metrics['std']:.4f})")
    logger.info(f"  Range: [{cv_metrics['min']:.4f}, {cv_metrics['max']:.4f}]")
    
    return cv_metrics
