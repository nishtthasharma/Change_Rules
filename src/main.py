import pandas as pd
import numpy as np
import time

from preprocessing import Preprocessor, levenshtein_distance
from config import DATASET_CONFIG
from context_beta import ContextBeta
from diffFunction import DifferentialFunction
from diffset import DiffSet
from changerulesdiscovery import ChangeRuleDiscovery
from predefinedDiff import PredefinedDiffFunction

def run_pipeline(dataset_name, output_file="pipeline_output.csv", diff_file=None, givenTheta = 0.8, verbose=False):
    config = DATASET_CONFIG[dataset_name]

    pipeline_start = time.perf_counter()

    pre = Preprocessor(config["file"], config["target_attribute"])
    df = pre.load_data()
    if verbose:
        print(pre.dataset_summary())
    delta_df = pre.compute_changes(config["target_flag"])

    # ---------------- Context Beta ----------------
    startTime_beta = time.perf_counter()
    context = ContextBeta(delta_df, config["target_attribute"], window_size=4)
    z_df = context.compute_z_scores()
    labels = context.generate_labels(z_df, config["target_flag"], threshold=1.5)
    weights = context.train_context_model(z_df, labels)
    beta_values = context.compute_beta(z_df, weights)
    augmented_df = context.augment_target_change(beta_values).dropna()
    endTime_beta = time.perf_counter()
    betaTime = endTime_beta - startTime_beta
    if verbose:
        print(f"Beta Calculation - Total Time: {betaTime:.4f} seconds")

    master_rules = []
    timing_records = []
    sort_index_cache = {}

    if verbose:
        print(df.columns)
    for ordering_attr in df.columns:
        if verbose:
            print(f"\n--- Ordering by {ordering_attr} ---")
        ordering_start = time.perf_counter()

        if ordering_attr == config["target_attribute"]:
            delta_df_ordered = augmented_df
        else:
            if ordering_attr not in sort_index_cache:
                sort_index_cache[ordering_attr] = np.argsort(
                    df[ordering_attr].to_numpy(),
                    kind="mergesort"
                )
            delta_df_ordered = compute_delta_from_order(df, sort_index_cache[ordering_attr])

        if verbose:
            print(f"\nChanges after ordering by {ordering_attr}")
            print(delta_df_ordered.head())

        # --- Differential Function Discovery ---
        # print(len(df_to_use), len(delta_df_ordered))
        startTime_diffFunc = time.perf_counter()

        ordering_attr_delta = f"delta_{ordering_attr}"

        if diff_file is not None:
            predef = PredefinedDiffFunction(diff_file, delta_df_ordered)
            singletons, bitsets = predef.build_for_ordering(ordering_attr=ordering_attr, theta=givenTheta)
            endTime_diffFunc = time.perf_counter()
            diffFuncTime = endTime_diffFunc - startTime_diffFunc
            if verbose:
                print(f"\nDiff Functions (Loaded) - Time: {diffFuncTime:.4f} seconds")
            if  not singletons:
                continue
        else:
            diffFunc = DifferentialFunction(delta_df_ordered)
            singletons, bitsets = diffFunc.discover(
                theta=givenTheta,
                priority_col=ordering_attr_delta,
                stop_if_missing=True,
            )
            endTime_diffFunc = time.perf_counter()
            diffFuncTime = endTime_diffFunc - startTime_diffFunc
            if verbose:
                print(f"\nDiff Functions (Discovered) - Time: {diffFuncTime:.4f} seconds")

        if not singletons or not singletons.get(ordering_attr_delta):
            if verbose:
                print(f"No diff functions for {ordering_attr_delta}; skipping rule discovery.")
            timing_records.append({
                "ordering_attr": ordering_attr,
                "diff_func_time": diffFuncTime,
                "change_rule_time": 0.0,
                "total_ordering_time": time.perf_counter() - ordering_start
            })
            continue

        # --- Change Rules Discovery ---
        cr_discover = ChangeRuleDiscovery(singletons, bitsets)
        startTime_cr = time.perf_counter()
        rules = cr_discover.discover_rules(lhs_attr=ordering_attr_delta, chi=0.7)
        endTime_cr = time.perf_counter()
        crTime = endTime_cr - startTime_cr
        if verbose:
            print(f"\nChange Rules Discovery - Time: {crTime:.4f} seconds")
            print(f"Rules discovered: {len(rules)}")

        master_rules.extend(rules)

        ordering_end = time.perf_counter()
        total_ordering_time = ordering_end - ordering_start

        timing_records.append({
            "ordering_attr": ordering_attr,
            "diff_func_time": diffFuncTime,
            "change_rule_time": crTime,
            "total_ordering_time": total_ordering_time
        })

    pipeline_end = time.perf_counter()
    total_pipeline_time = pipeline_end - pipeline_start
    print(f"\nTotal change rules discovered across all attributes: {len(master_rules)}")
    print(f"Total pipeline runtime: {total_pipeline_time:.4f} seconds")

    rules_df = pd.DataFrame(master_rules)
    if not rules_df.empty:
        rules_df = rules_df[["lhs_attr", "lhs_interval", "rhs_attr", "rhs_interval", "confidence", "support_global"]]

    timing_df = pd.DataFrame(timing_records)
    
    timing_df.loc["TOTAL"] = {
        "ordering_attr": "TOTAL_PIPELINE",
        "diff_func_time": timing_df["diff_func_time"].sum(),
        "change_rule_time": timing_df["change_rule_time"].sum(),
        "total_ordering_time": total_pipeline_time
    }

    with open(output_file, "w") as f:
        f.write("Rules\n")
        rules_df.to_csv(f, index=False)
        f.write("\nTiming\n")
        timing_df.to_csv(f, index=False)

    print(f"All rules and runtimes saved to {output_file}")

def compute_delta(df, abs_delta=False):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    delta_data = {}

    if numeric_cols:
        numeric_deltas = df[numeric_cols].diff()
        if abs_delta:
            numeric_deltas = numeric_deltas.abs()
        numeric_deltas = numeric_deltas.iloc[1:].reset_index(drop=True)
        for col in numeric_cols:
            delta_data[f"delta_{col}"] = numeric_deltas[col]

    for col in non_numeric_cols:
        values = df[col].astype(str).to_numpy()
        changes = np.array(
            [levenshtein_distance(v1, v2) for v1, v2 in zip(values[:-1], values[1:])],
            dtype=float,
        )
        if abs_delta:
            changes = np.abs(changes)
        delta_data[f"delta_{col}"] = changes

    return pd.DataFrame(delta_data)


def compute_delta_from_order(df, order, abs_delta=False):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    delta_data = {}

    if numeric_cols:
        ordered_numeric = df[numeric_cols].to_numpy()[order]
        numeric_deltas = np.diff(ordered_numeric, axis=0)
        if abs_delta:
            numeric_deltas = np.abs(numeric_deltas)
        for idx, col in enumerate(numeric_cols):
            delta_data[f"delta_{col}"] = numeric_deltas[:, idx]

    for col in non_numeric_cols:
        values = df[col].astype(str).to_numpy()[order]
        changes = np.array(
            [levenshtein_distance(v1, v2) for v1, v2 in zip(values[:-1], values[1:])],
            dtype=float,
        )
        if abs_delta:
            changes = np.abs(changes)
        delta_data[f"delta_{col}"] = changes

    return pd.DataFrame(delta_data)


if __name__ == "__main__":
    dataset = "dataset1"
    run_pipeline(dataset, givenTheta=0.6)
