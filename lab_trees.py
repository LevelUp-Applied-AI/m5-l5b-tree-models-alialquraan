"""
Module 5 Week B — Lab: Trees & Ensembles

Build and evaluate decision tree and random forest models on the
Petra Telecom churn dataset with class imbalance handling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, precision_recall_curve,
                             average_precision_score, PrecisionRecallDisplay)
import matplotlib.pyplot as plt


NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents", "contract_months"]


def load_and_split(filepath="data/telecom_churn.csv", random_state=42):
    """Load data and split into train/test with stratification.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # TODO: Load CSV, select numeric features, split with stratification
    pass


def build_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    """Train a DecisionTreeClassifier.

    Returns:
        Fitted DecisionTreeClassifier.
    """
    # TODO: Create and fit the model
    pass


def build_random_forest(X_train, y_train, n_estimators=100,
                        class_weight=None, random_state=42):
    """Train a RandomForestClassifier.

    Args:
        class_weight: None for default, 'balanced' for imbalance handling.

    Returns:
        Fitted RandomForestClassifier.
    """
    # TODO: Create and fit the model with the given class_weight
    pass


def get_feature_importances(model, feature_names):
    """Extract and sort feature importances.

    Returns:
        Dictionary mapping feature name to importance, sorted descending.
    """
    # TODO: Extract feature_importances_ and return sorted dict
    pass


def compute_pr_auc(model, X_test, y_test):
    """Compute PR-AUC (average precision) for the model.

    Returns:
        Float: average precision score.
    """
    # TODO: Use predict_proba to get probabilities, then compute average_precision_score
    pass


if __name__ == "__main__":
    result = load_and_split()
    if result:
        X_train, X_test, y_train, y_test = result
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"Churn rate: {y_train.mean():.2%}")

        # Task 2: Decision tree
        tree = build_decision_tree(X_train, y_train)
        if tree:
            print(f"\nDecision Tree: depth={tree.get_depth()}")
            print(classification_report(y_test, tree.predict(X_test)))

        # Task 3: Random forest
        rf = build_random_forest(X_train, y_train)
        if rf:
            importances = get_feature_importances(rf, NUMERIC_FEATURES)
            if importances:
                print(f"Top 5 features: {dict(list(importances.items())[:5])}")

        # Task 4: Balanced models
        rf_balanced = build_random_forest(X_train, y_train, class_weight="balanced")
        if rf_balanced:
            print("\nBalanced Random Forest:")
            print(classification_report(y_test, rf_balanced.predict(X_test)))

        # Task 5: PR-AUC
        if rf and rf_balanced:
            auc_default = compute_pr_auc(rf, X_test, y_test)
            auc_balanced = compute_pr_auc(rf_balanced, X_test, y_test)
            if auc_default and auc_balanced:
                print(f"PR-AUC (default): {auc_default:.3f}")
                print(f"PR-AUC (balanced): {auc_balanced:.3f}")
