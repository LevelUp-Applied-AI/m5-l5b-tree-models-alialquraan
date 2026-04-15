"""Autograder tests for Lab 5B — Trees & Ensembles."""

import pytest
import sys
import os
from sklearn.metrics import recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lab_trees import (load_and_split, build_decision_tree, build_random_forest,
                       get_feature_importances, compute_pr_auc, NUMERIC_FEATURES)


@pytest.fixture
def data():
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    result = load_and_split()
    assert result is not None, "load_and_split returned None"
    return result


# ── Data Splitting ────────────────────────────────────────────────────────

def test_data_split(data):
    """Split should produce ~80/20 train/test ratio."""
    X_train, X_test, y_train, y_test = data
    total = len(X_train) + len(X_test)
    assert total > 1000
    test_ratio = len(X_test) / total
    assert 0.18 <= test_ratio <= 0.22


def test_split_stratification(data):
    """Class distribution should be preserved in both splits."""
    X_train, X_test, y_train, y_test = data
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.03, (
        f"Stratification not preserved: train={train_rate:.3f}, test={test_rate:.3f}"
    )


# ── Decision Tree ─────────────────────────────────────────────────────────

def test_decision_tree_exists(data):
    """Decision tree should be fitted with max_depth=5."""
    X_train, X_test, y_train, y_test = data
    model = build_decision_tree(X_train, y_train)
    assert model is not None
    from sklearn.tree import DecisionTreeClassifier
    assert isinstance(model, DecisionTreeClassifier)
    assert model.max_depth == 5
    assert hasattr(model, "classes_")


def test_decision_tree_predictions(data):
    """Decision tree should produce valid binary predictions."""
    X_train, X_test, y_train, y_test = data
    model = build_decision_tree(X_train, y_train)
    assert model is not None
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1}), "Predictions should be binary (0/1)"


# ── Random Forest ─────────────────────────────────────────────────────────

def test_random_forest_exists(data):
    """Random forest should be fitted with 100 estimators."""
    X_train, X_test, y_train, y_test = data
    model = build_random_forest(X_train, y_train)
    assert model is not None
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert hasattr(model, "classes_")


# ── Feature Importances ──────────────────────────────────────────────────

def test_feature_importances_extracted(data):
    """Importances should be a sorted dict summing to ~1.0."""
    X_train, X_test, y_train, y_test = data
    model = build_random_forest(X_train, y_train)
    assert model is not None
    importances = get_feature_importances(model, NUMERIC_FEATURES)
    assert importances is not None
    assert len(importances) == len(NUMERIC_FEATURES)
    total = sum(importances.values())
    assert abs(total - 1.0) < 0.01
    values = list(importances.values())
    assert values == sorted(values, reverse=True)


# ── Class Imbalance Handling ──────────────────────────────────────────────

def test_balanced_recall_improvement(data):
    """Balanced class weights should improve recall on the positive class."""
    X_train, X_test, y_train, y_test = data
    rf_default = build_random_forest(X_train, y_train)
    rf_balanced = build_random_forest(X_train, y_train, class_weight="balanced")
    assert rf_default is not None and rf_balanced is not None
    recall_default = recall_score(y_test, rf_default.predict(X_test))
    recall_balanced = recall_score(y_test, rf_balanced.predict(X_test))
    assert recall_balanced > recall_default, \
        f"Balanced recall ({recall_balanced:.3f}) should exceed default ({recall_default:.3f})"


# ── PR-AUC ────────────────────────────────────────────────────────────────

def test_pr_auc_values(data):
    """PR-AUC should be a valid float in (0, 1]."""
    X_train, X_test, y_train, y_test = data
    rf = build_random_forest(X_train, y_train)
    rf_balanced = build_random_forest(X_train, y_train, class_weight="balanced")
    assert rf is not None and rf_balanced is not None
    auc1 = compute_pr_auc(rf, X_test, y_test)
    auc2 = compute_pr_auc(rf_balanced, X_test, y_test)
    assert auc1 is not None and auc2 is not None
    assert 0 < auc1 <= 1, f"PR-AUC should be in (0, 1], got {auc1}"
    assert 0 < auc2 <= 1, f"PR-AUC should be in (0, 1], got {auc2}"
    assert auc2 > 0.3, f"Balanced PR-AUC should be > 0.3, got {auc2}"


def test_pr_auc_uses_probabilities(data):
    """PR-AUC computation should use predict_proba, not predict."""
    X_train, X_test, y_train, y_test = data
    rf = build_random_forest(X_train, y_train)
    assert rf is not None
    auc = compute_pr_auc(rf, X_test, y_test)
    assert auc is not None
    # predict_proba-based PR-AUC should differ from accuracy
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, rf.predict(X_test))
    assert auc != acc, (
        "PR-AUC should not equal accuracy — are you using predict_proba?"
    )
