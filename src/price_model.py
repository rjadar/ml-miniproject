"""Reference collectible-price classification pipeline."""

import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)


PRICE_BIN_COUNT = 5
NUMERIC_FEATURES = ["search_volume_normalized"]
CATEGORICAL_FEATURES = ["category", "material", "special_version"]


def load_and_label_data(csv_path):
    """Load the curated data and reproduce the notebook's target/features."""
    df = pd.read_csv(csv_path)
    prices = np.asarray(df["average_sale_price"]).reshape(-1, 1)
    price_bins = (
        np.exp(np.log(np.max(prices) / np.min(prices)) / PRICE_BIN_COUNT
               * np.array([0, 1, 2, 3, 4, 5]))
        * np.min(prices)
    )

    def get_price_bin(row):
        price = row.average_sale_price
        for index in range(PRICE_BIN_COUNT):
            if price >= price_bins[index] and price < price_bins[index + 1]:
                return index + 1
        return None

    df["price_bin"] = df.apply(get_price_bin, axis=1)
    log_volume = np.log(df["search_volume"])
    log_vol_min = np.min(log_volume)
    log_vol_max = np.max(log_volume)
    df["search_volume_normalized"] = df["search_volume"].apply(
        lambda value: (math.log(value) - log_vol_min) / (log_vol_max - log_vol_min)
    )
    return df


def make_one_hot_features(df):
    """Fit the reference preprocessing on the full data before CV."""
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    X = preprocessor.fit_transform(df)
    y = df["price_bin"].to_numpy()
    return X, y


def make_ordinal_features(df):
    """Build the reference ordinal representation and category probabilities."""
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[("ordinal", OrdinalEncoder())]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    X = preprocessor.fit_transform(df)
    probabilities = []
    for column in (1, 2, 3):
        count = int(np.max(X[:, column])) + 1
        probabilities.append(
            np.array([np.mean(X[:, column] == value) for value in range(count)])
        )
    return X, probabilities


def make_mixed_feature_kernel(probabilities):
    """Return the notebook's weighted RBF/categorical precomputed kernel."""
    theta = 0.2
    alpha = 0.5
    feature_weights = (2, 1, 1, 3)
    categorical_similarity = [
        np.power(1 - np.power(probability, alpha), 1 / alpha)
        for probability in probabilities
    ]

    def kernel(X1, X2):
        search_delta = X1[:, 0, None] - X2[None, :, 0]
        search_similarity = np.exp(-(search_delta ** 2) / (2 * theta ** 2))
        similarities = [search_similarity]
        for position, scores in zip((1, 2, 3), categorical_similarity):
            values_1 = X1[:, position].astype(int)
            values_2 = X2[:, position].astype(int)
            matches = values_1[:, None] == values_2[None, :]
            similarities.append(np.where(matches, scores[values_1][:, None], 0))
        return sum(weight * similarity for weight, similarity
                   in zip(feature_weights, similarities))

    return kernel


def _score(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "class_precision": precision_score(
            y_true, y_pred, average=None, zero_division=0
        ),
    }


def cross_validate_model(X, y, model_factory, kernel=None):
    """Reproduce four-fold train/test weighted and per-class precision metrics."""
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    train_scores = []
    test_scores = []
    for train_indices, test_indices in splitter.split(X, y):
        train_X, test_X = X[train_indices], X[test_indices]
        train_y, test_y = y[train_indices], y[test_indices]
        if kernel is not None:
            fit_X = kernel(train_X, train_X)
            test_X = kernel(test_X, train_X)
        else:
            fit_X = train_X
        model = model_factory()
        model.fit(fit_X, train_y)
        train_prediction = model.predict(fit_X)
        test_prediction = model.predict(test_X)
        train_scores.append(_score(train_y, train_prediction))
        test_scores.append(_score(test_y, test_prediction))

    result = {}
    for split_name, scores in (("train", train_scores), ("test", test_scores)):
        for metric in ("precision", "recall", "f1"):
            result[f"{split_name}_{metric}"] = float(
                np.mean([score[metric] for score in scores])
            )
        result[f"{split_name}_class_precision"] = np.mean(
            [score["class_precision"] for score in scores], axis=0
        )
    return result


def print_summary(name, result):
    print(f"\n{name}")
    print("Train weighted P/R/F1: " + " / ".join(
        f"{result['train_' + metric]:.3f}" for metric in ("precision", "recall", "f1")
    ))
    print("Test  weighted P/R/F1: " + " / ".join(
        f"{result['test_' + metric]:.3f}" for metric in ("precision", "recall", "f1")
    ))
    print("Train per-class precision: " + np.array2string(
        result["train_class_precision"], precision=3
    ))
    print("Test  per-class precision: " + np.array2string(
        result["test_class_precision"], precision=3
    ))


def _make_tuning_pipeline(model):
    """Preprocess raw columns inside each GridSearchCV training fold."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
            ("minmax", MinMaxScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ["search_volume"]),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def _report_learned_parameters(name, best_estimator):
    """Show learned SVM support vectors or decision-tree split rules."""
    model = best_estimator.named_steps["model"]
    if hasattr(model, "support_"):
        print(
            "  Learned SVM parameters (fold 1 training refit): "
            f"{len(model.support_)} support vectors; "
            f"{model.n_support_.tolist()} per class"
        )
    elif hasattr(model, "tree_"):
        tree = model.tree_
        feature_names = best_estimator.named_steps[
            "preprocess"
        ].get_feature_names_out()
        rules = []
        for node, feature in enumerate(tree.feature):
            if feature >= 0:
                rules.append(
                    f"{feature_names[feature]} <= {tree.threshold[node]:.4g}"
                )
            if len(rules) == 6:
                break
        print(
            "  Learned decision-tree parameters (fold 1 training refit): "
            f"depth={tree.max_depth}, nodes={tree.node_count}; "
            "split rules: " + ("; ".join(rules) if rules else "leaf only")
        )


def run_nested_tuning(df, y, csv_path):
    """Tune on inner stratified folds; evaluate only on untouched outer folds."""
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    experiments = [
        (
            "linear_svm",
            SVC(kernel="linear", decision_function_shape="ovo"),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        (
            "rbf_svm",
            SVC(kernel="rbf", decision_function_shape="ovo"),
            {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", 0.1, 1.0]},
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(random_state=1),
            {
                "model__max_depth": [3, None],
                "model__min_samples_split": [2, 10],
                "model__min_samples_leaf": [1, 5],
            },
        ),
    ]
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    # Each outer training fold has only three examples from the rarest class.
    # Three inner folds preserve stratification without empty class folds.
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    result_rows = []

    print("\nNested-CV hyperparameter tuning (inner CV selects; outer fold evaluates)")
    print("Inner tuning CV: 3 stratified folds (smallest outer-training class has 3 samples); outer evaluation remains 4 folds.")
    for name, estimator, param_grid in experiments:
        configurations = list(ParameterGrid(param_grid))
        print(f"\n{name}: tested {len(configurations)} configurations: {configurations}")
        outer_scores = []
        for fold, (train_indices, test_indices) in enumerate(outer_cv.split(df, y), 1):
            search = GridSearchCV(
                _make_tuning_pipeline(estimator),
                param_grid=param_grid,
                scoring="f1_weighted",
                cv=inner_cv,
                n_jobs=1,
                refit=True,
            )
            train_df = df.iloc[train_indices]
            train_y = y[train_indices]
            test_df = df.iloc[test_indices]
            test_y = y[test_indices]
            search.fit(train_df, train_y)
            prediction = search.predict(test_df)
            scores = _score(test_y, prediction)
            outer_scores.append(scores)
            print(
                f"  Fold {fold}: best_params={search.best_params_}; "
                f"inner-CV weighted F1={search.best_score_:.3f}; "
                f"outer test P/R/F1="
                f"{scores['precision']:.3f}/{scores['recall']:.3f}/{scores['f1']:.3f}"
            )
            if fold == 1:
                _report_learned_parameters(name, search.best_estimator_)
            row = {
                "model": name,
                "outer_fold": fold,
                "tested_configurations": len(configurations),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "inner_cv_weighted_f1": search.best_score_,
                "outer_test_precision": scores["precision"],
                "outer_test_recall": scores["recall"],
                "outer_test_f1": scores["f1"],
            }
            row.update({
                f"outer_test_bin_{bin_number}_precision": value
                for bin_number, value in enumerate(scores["class_precision"], 1)
            })
            result_rows.append(row)

        print(
            "  Final outer-CV weighted P/R/F1: "
            + " / ".join(
                f"{np.mean([score[metric] for score in outer_scores]):.3f}"
                for metric in ("precision", "recall", "f1")
            )
        )
        print(
            "  Final outer-CV per-class precision: "
            + np.array2string(
                np.mean([score["class_precision"] for score in outer_scores], axis=0),
                precision=3,
            )
        )

    results_path = Path(csv_path)
    pd.DataFrame(result_rows).to_csv(results_path, index=False)
    print(f"\nTuning results saved to {results_path}")
