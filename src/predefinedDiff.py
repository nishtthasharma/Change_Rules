import numpy as np
import math
import time

class PredefinedDiffFunction:
    def __init__(self, file_path, delta_df):
        self.file_path = file_path
        self.df = delta_df.reset_index(drop=True)
        self.delta_cols = list(self.df.columns)

        self.Psi_by_ordering = self._parse_file()

    def _parse_file(self):
        Psi_by_ordering = {}
        current_ordering = None

        with open(self.file_path, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                # Detect ordering attribute
                if line.endswith("-"):
                    current_ordering = line.replace("-", "").strip()
                    Psi_by_ordering[current_ordering] = {}
                    continue

                # Parse attribute intervals
                attr, intervals_str = line.split(":")
                attr = attr.strip()

                intervals = []
                for part in intervals_str.strip().split(";"):
                    part = part.strip().replace("[", "").replace("]", "")
                    l, u = map(float, part.split(","))
                    intervals.append((l, u))

                Psi_by_ordering[current_ordering][f"delta_{attr}"] = intervals

        return Psi_by_ordering

    def longest_consecutive_segment(self, bitset):
        if not np.any(bitset):
            return 0, 0

        padded = np.concatenate(([False], bitset, [False])).astype(np.int8)
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        lengths = ends - starts

        best_idx = int(np.argmax(lengths))
        return int(lengths[best_idx]), int(starts[best_idx])

    def build_for_ordering(self, ordering_attr, theta=0.1):
        Psi_filtered = {}
        Bitsets = {}
        
        if ordering_attr not in self.Psi_by_ordering:
            print(f"No diff functions found for ordering: {ordering_attr}")
            return Psi_filtered, Bitsets

        Psi_raw = self.Psi_by_ordering[ordering_attr]

        m = len(self.df)
        tau = math.ceil(theta * m)

        print(f"\n[Predefined] Processing ordering: {ordering_attr}")
        print(f"Minimum segment length (tau): {tau}")

        for col, intervals in Psi_raw.items():
            if col not in self.df.columns:
                print(f"Warning: {col} not found in delta_df. Skipping.")
                continue

            values = self.df[col].values

            Psi_filtered[col] = []
            Bitsets[col] = {}

            for (l, u) in intervals:
                startBitset = time.perf_counter()
                bitset = (values >= l) & (values <= u)
                endBitset = time.perf_counter()
                totalTimeBitset = endBitset - startBitset
                print(f"Time taken to calculate bitset for interval {(l,u)}: {totalTimeBitset} seconds.")

                max_run, start_idx = self.longest_consecutive_segment(bitset)

                if max_run >= tau:
                    Psi_filtered[col].append({
                        "attribute": col,
                        "interval": (l, u),
                        "support": max_run / m,
                        "max_seg": max_run,
                        "start_idx": start_idx
                    })

                    Bitsets[col][(l, u)] = bitset

                endInterval = time.perf_counter()
                totalIntervalTime = endInterval - startBitset
                print(f"Total time for interval {(l,u)}: {totalIntervalTime} seconds.")

        return Psi_filtered, Bitsets
