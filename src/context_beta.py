import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

class ContextBeta:
    def __init__(self, delta_df, target_attribute, window_size = 4):
        self.df = delta_df
        self.target = f"delta_{target_attribute}"
        self.window = window_size
    
    def compute_z_scores(self):
        print("\nComputing z-scores for window of size", self.window)
        delta_cols = [c for c in self.df.columns if c.startswith("delta_")]
        delta_only = self.df[delta_cols]

        epsilon = 1e-8
        rolling_mean = delta_only.rolling(self.window).mean().shift(1)
        rolling_std = delta_only.rolling(self.window).std().shift(1)
        z_df = ((delta_only - rolling_mean) / (rolling_std + epsilon)).clip(-5, 5)
        return z_df
    
    def train_context_model(self, z_df, labels):
        z_df = z_df.iloc[self.window:]
        labels = labels.iloc[self.window:]

        X = z_df.drop(columns=[self.target])
        y = labels

        print("\nTraining RF Model")

        if X.empty:
            return {}

        X = X.fillna(0.0).astype(np.float32)
        y = y.astype(np.int8)

        if y.nunique() < 2:
            return {col: 0.0 for col in X.columns}

        X_train, y_train = self._sample_training_data(X, y)

        model = RandomForestClassifier(
            n_estimators=64,
            max_depth=12,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        importances = model.feature_importances_
        weights = dict(zip(X.columns, importances))

        # metrics = evaluate_rf_model(model, X, y)

        return weights

    def _sample_training_data(self, X, y, max_rows=6000):
        if len(X) <= max_rows:
            return X, y

        rng = np.random.default_rng(42)
        y_np = y.to_numpy()
        class_values = np.unique(y_np)

        selected_parts = []
        per_class = max(max_rows // len(class_values), 1)

        for class_value in class_values:
            class_indices = np.flatnonzero(y_np == class_value)
            take = min(len(class_indices), per_class)
            chosen = rng.choice(class_indices, size=take, replace=False)
            selected_parts.append(chosen)

        selected = np.sort(np.concatenate(selected_parts))
        return X.iloc[selected], y.iloc[selected]

    
    def compute_beta(self, z_df, weights, k = 2):

        print("\nComputing beta")
        target_col = self.target

        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_k_features = [f for f, _ in sorted_weights[:k]]

        print("Top context features:", top_k_features)

        z_valid = z_df.iloc[self.window:]
        if not top_k_features:
            return z_valid[target_col].to_numpy()

        weight_vector = np.array([weights[col] for col in top_k_features], dtype=float)
        weighted_sum = z_valid[top_k_features].to_numpy() @ weight_vector
        beta_values = z_valid[target_col].to_numpy() - weighted_sum
        return beta_values
    
    def augment_target_change(self, beta_values):
        target_vals = self.df[self.target].iloc[self.window:].to_numpy()
        augmented_df = self.df.iloc[self.window:].copy()
        augmented_df[self.target] = target_vals * np.asarray(beta_values)
        return augmented_df
    
    def generate_labels(self, z_df, target_flag = "", threshold = 1.5, lambda_factor = 1.5):

        print("\nGenerating labels")
        target_col = self.target
        if target_flag and target_flag in self.df.columns:
            labels = self.df[target_flag].iloc[1:].reset_index(drop=True)
            return labels
        else:
            context_cols = [c for c in z_df.columns if c != target_col]
            if not context_cols:
                return (z_df[target_col] > threshold).astype(int)

            context_median = z_df[context_cols].median(axis=1)
            labels = (z_df[target_col] > threshold) | (
                z_df[target_col] > (lambda_factor * context_median)
            )
            return labels.astype(int)

def evaluate_rf_model(model, X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify = y)

    y_pred = model.predict(X_test)

    print("=== Random Forest Evaluation ===")
    print(classification_report(y_test, y_pred, digits=4))
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    metrics_dict = {
        "accuracy": accuracy,
        "f1_score": f1
    }
    
    return metrics_dict
