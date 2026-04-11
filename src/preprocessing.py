import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self, file_path, target_attribute):
        self.file_path = file_path
        self.target_attribute = target_attribute
        self.df = None
    
    def load_data(self):
        print("Loading file: ", self.file_path)
        self.df = pd.read_csv(self.file_path)
        return self.df
    
    def dataset_summary(self):
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = self.df.select_dtypes(exclude=['number']).columns.tolist()

        summary = {
            "Total Tuples": len(self.df),
            "Total Attributes": len(self.df.columns),
            "Num Numeric": len(numeric_cols),
            "Num Non_numeric": len(non_numeric_cols),
            "Numeric Attributes": numeric_cols,
            "Non_numeric Attributes": non_numeric_cols
        }
        return summary
    
    def order_by_target(self):
        print("\nOrdering by: ", self.target_attribute)
        self.df = self.df.sort_values(by=self.target_attribute).reset_index(drop=True)
        return self.df
    
    def compute_changes(self, target_flag=""):
        print("\nComputing consecutive changes")
        delta_df = self.df.copy()
        numeric_cols = delta_df.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = delta_df.select_dtypes(exclude=['number']).columns.tolist()

        if target_flag and target_flag in numeric_cols:
            numeric_cols.remove(target_flag)

        if target_flag and target_flag in non_numeric_cols:
            non_numeric_cols.remove(target_flag)

        if numeric_cols:
            numeric_deltas = delta_df[numeric_cols].diff().round(5)
            numeric_deltas = numeric_deltas.mask(np.isclose(numeric_deltas, 0), 0.0)
            numeric_deltas.columns = [f"delta_{col}" for col in numeric_cols]
            delta_df[numeric_deltas.columns] = numeric_deltas

        for col in non_numeric_cols:
            values = delta_df[col].astype(str).to_numpy()
            changes = [None]
            changes.extend(
                levenshtein_distance(v1, v2)
                for v1, v2 in zip(values[:-1], values[1:])
            )
            delta_df[f"delta_{col}"] = changes

        delta_df = delta_df.iloc[1:].reset_index(drop=True)
        return delta_df
    
def levenshtein_distance(s1, s2):
    s1, s2 = str(s1), str(s2)
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):

            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i-1][j] + 1,      
                dp[i][j-1] + 1,      
                dp[i-1][j-1] + cost  
            )
    return dp[m][n] 
