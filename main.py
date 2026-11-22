
"""Run the fast presentation demo or the complete reference experiment."""

import argparse
import time
from pathlib import Path

_START_TIME = time.perf_counter()

from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.price_model import (
    cross_validate_model,
    load_and_label_data,
    make_mixed_feature_kernel,
    make_one_hot_features,
    make_ordinal_features,
    print_summary,
    run_nested_tuning,
)


def run_demo(X, y):
    """Compare the original and class-balanced Linear SVM."""
    print("Fast demo: Linear SVM, four-fold stratified cross-validation")

    baseline = cross_validate_model(
        X,
        y,
        lambda: SVC(kernel="linear", decision_function_shape="ovo"),
    )
    print_summary("Linear SVM (one-vs-one) baseline", baseline)

    balanced = cross_validate_model(
        X,
        y,
        lambda: SVC(
            kernel="linear",
            decision_function_shape="ovo",
            class_weight="balanced",
        ),
    )
    print_summary("Class-balanced Linear SVM (one-vs-one)", balanced)


def run_full_experiment(df, X, y):
    """Run reference models, class-balanced SVM, and bagging experiments."""
    print("Full experiment: four-fold stratified cross-validation")

    models = [
        (
            "Linear SVM (one-vs-rest)",
            X,
            lambda: OneVsRestClassifier(SVC(kernel="linear")),
            None,
        ),
        (
            "Linear SVM (one-vs-one)",
            X,
            lambda: SVC(kernel="linear", decision_function_shape="ovo"),
            None,
        ),
        (
            "Class-balanced Linear SVM (one-vs-one)",
            X,
            lambda: SVC(
                kernel="linear",
                decision_function_shape="ovo",
                class_weight="balanced",
            ),
            None,
        ),
        (
            "RBF SVM (one-vs-one)",
            X,
            lambda: SVC(kernel="rbf", decision_function_shape="ovo"),
            None,
        ),
    ]

    for name, features, factory, kernel in models:
        result = cross_validate_model(features, y, factory, kernel)
        print_summary(name, result)

    # Original custom-kernel SVM
    ordinal_X, probabilities = make_ordinal_features(df)
    custom_kernel = make_mixed_feature_kernel(probabilities)

    custom_result = cross_validate_model(
        ordinal_X,
        y,
        lambda: SVC(
            kernel="precomputed",
            decision_function_shape="ovo",
        ),
        kernel=custom_kernel,
    )
    print_summary("Custom-kernel SVM", custom_result)

    # Original decision tree
    tree_result = cross_validate_model(
        X,
        y,
        lambda: DecisionTreeClassifier(random_state=1),
    )
    print_summary("Decision tree", tree_result)

    # Original bagging experiment
    bagging_rows = []

    for tree_count in (2, 5, 10, 20, 50, 100, 200):
        result = cross_validate_model(
            X,
            y,
            lambda count=tree_count: BaggingClassifier(
                estimator=DecisionTreeClassifier(),
                n_estimators=count,
                random_state=1,
            ),
        )

        row = {"num_trees": tree_count}

        for split in ("train", "test"):
            for metric in ("precision", "recall", "f1"):
                row[f"avg_{split}_{metric}"] = result[
                    f"{split}_{metric}"
                ]

            for bin_number, value in enumerate(
                result[f"{split}_class_precision"],
                start=1,
            ):
                row[f"class_{bin_number}_{split}_prec"] = value

        bagging_rows.append(row)

    import pandas as pd

    print("\nBagged decision tree results:")
    print(
        pd.DataFrame(bagging_rows)
        .set_index("num_trees")
        .to_string(float_format="%.3f")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--full",
        action="store_true",
        help="run all reference models, balanced SVM, and bagging experiments",
    )

    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "src" / "data_sourcing" / "data_curated.csv"

    df = load_and_label_data(data_path)
    X, y = make_one_hot_features(df)

    if args.full:
        run_full_experiment(df, X, y)
    else:
        run_demo(X, y)

    # Preserve the existing nested-CV tuning experiment.
    run_nested_tuning(df, y, project_dir / "tuning_results.csv")

    print(f"\nTotal runtime: {time.perf_counter() - _START_TIME:.2f} seconds")


if __name__ == "__main__":
    main()
