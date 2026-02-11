from joblib import Parallel, delayed
import pandas as pd
from multiprocessing import Pool
import numpy as np
from collections import defaultdict

class AHPWeights:
    def __init__(self, df=None, place_level="all", n_jobs=4, parallel=True):
        self.place_level = place_level
        self.N_JOBS = n_jobs
        
        if df is not None:
            if place_level.lower()=="all":
                self.comparisons_df = df
            elif place_level.lower()=="city":
                self.comparisons_df = df[df["left_city"]==df["right_city"]].copy()
            elif place_level.lower()=="country":
                self.comparisons_df = df[df["left_country"]==df["right_country"]].copy()
            elif place_level.lower()=="continent":
                self.comparisons_df = df[df["left_continent"]==df["right_continent"]].copy()
            else:
                self.comparisons_df = df
            
    def get_matches(self):
        return self.matches_df
            
    def get_metrics(self):
        return self.comparisons_df["category"].unique()
        
    ######################
    #  AHP functions     #
    ######################
    
    def filter_player(self, cat_df, player="left"):
        df_ = cat_df[[f"{player}_id", "winner", "category", f"{player}_lat", f"{player}_long", f"{player}_city", f"{player}_country", f"{player}_continent"]].copy()
        
        df_.rename(columns={f"{player}_id": "image_id", f"{player}_lat": "lat", f"{player}_long":"long", f"{player}_city":"city", f"{player}_country":"country", f"{player}_continent":"continent"}, inplace=True)

        df_.drop(columns=["winner", "category"], inplace=True)
        df_.sort_values(by=["image_id"], inplace=True)
        
        return df_
    
    def prepare_matches(self, metric="safety"):
        
        df_ = self.comparisons_df[self.comparisons_df["category"]==metric].copy()
        self.matches_df = df_
        
        
        left_df = self.filter_player(df_, player="left")
        right_df = self.filter_player(df_, player="right")
        
        images_df = pd.concat([left_df, right_df], ignore_index=True)
        images_df.drop_duplicates(inplace=True)
        images_df.sort_values(by=["image_id"], inplace=True)
        
        self.image_ids = np.sort(images_df["image_id"].unique()).tolist()
        self.image_to_idx = {img: idx for idx, img in enumerate(self.image_ids)}
        self.images_df = images_df
        
    
    def calculate_ahp_weights(self, metric="safety", method="dict"):
        
        self.prepare_matches(metric=metric)
        df_ = self.get_matches().copy()
        print("Analyzing", df_.shape[0], metric, "comparisons")
        
        self.create_from_comparisons(df_, method=method)
        self.build_sparse_ahp(method=method)
        self.calculate_priority_vector(method=method)
            
    def normalize_ahp_weights(self, normalize=True, min_range=0, max_range=10, epsilon=0.0):
    
        weights_df = pd.DataFrame(data={"image_id": self.image_ids, "AHPweight": self.priority_vector.tolist()})
        #weights_df = pd.DataFrame(list(self.current_ratings.items()), columns=["image_id", "AHPweight"])
        
        if normalize:
        
            min_range = min_range + epsilon
            max_range = max_range - epsilon
        
            # Normalize Elo ratings to the range [0, 10]
            min_rating = weights_df["AHPweight"].min()
            max_rating = weights_df["AHPweight"].max()
            
            weights_df["AHPScore"] = min_range + ((weights_df["AHPweight"] - min_rating) / (max_rating - min_rating)) * (max_range - min_range)
            
        weights_df = pd.merge(weights_df, self.images_df, on="image_id", how="left")
        self.weights_df = weights_df
        
    def log_scaled(priority_vector, scale_min=0, scale_max=10):
        log_priorities = np.log(priority_vector + 1e-8)  # avoid log(0)
        min_val = np.min(log_priorities)
        max_val = np.max(log_priorities)

        scaled = (log_priorities - min_val) / (max_val - min_val)
        return scaled * (scale_max - scale_min) + scale_min
    
    def get_ahp_weights(self):
        return self.weights_df

    ####################
    # General function #
    ####################
    
    def create_from_comparisons(self, df_, method="dict"):
        if method == "dict":
            self.create_dict_from_comparisons(df_)
            
        elif method == "matrix":
            self.create_matrix_from_comparisons(df_)
            
        elif method == "dataframe":
            self.create_df_from_comparisons(df_)
            
        else:
            self.create_df_from_comparisons(df_)

    def build_sparse_ahp(self, method="dict"):
        if method == "dict":
            self.build_sparse_ahp_from_dict()
            
        elif method == "matrix":
            self.build_sparse_ahp_from_matrix()
            
        elif method == "dataframe":
            self.build_sparse_ahp_from_df()
            
        else:
            self.build_sparse_ahp_from_df()

    def calculate_priority_vector(self, method="dict"):
        if method == "dict":
            self.calculate_priority_vector_from_dict()
            self.priority_vector = self.priority_vector_dict
            
        elif method == "matrix":
            self.calculate_priority_vector_from_matrix()
            self.priority_vector = self.priority_vector_matrix
            
        elif method == "dataframe":
            self.calculate_priority_vector_from_df()
            self.priority_vector = self.priority_vector_df
            
        else:
            self.calculate_priority_vector_from_df()
            self.priority_vector = self.priority_vector_df

    ###################
    #  Dict function  #
    ###################
    
    def create_dict_from_comparisons(self, df_):
        
        self.votes_dict = defaultdict(lambda: [0, 0])
        
        for _, row in df_.iterrows():
            image_1, image_2, result = row["left_id"], row["right_id"], row["winner"]
            
            left = self.image_to_idx[row['left_id']]
            right = self.image_to_idx[row['right_id']]
            result = row['winner'].strip().lower()

            if result == 'left':
                self.votes_dict[(left, right)][0] += 1
            elif result == 'right':
                self.votes_dict[(right, left)][0] += 1
            elif result == 'equal':
                self.votes_dict[(left, right)][1] += 1
                self.votes_dict[(right, left)][1] += 1
    
    def build_sparse_ahp_from_dict(self):
        n = len(self.image_ids)
        ahp_dict = defaultdict(lambda: [1, 1])

        for (i, j), num in self.votes_dict.items():
            if i == j:
                continue
            den = self.votes_dict.get((j, i), [0, 0])
            numerator = num[0] + 0.5 * num[1]
            denominator = den[0] + 0.5 * num[1]
            value = numerator / denominator if denominator != 0 else 1e6
            ahp_dict[(i, j)] = value

        self.ahp_dict = ahp_dict
        
    def calculate_priority_vector_from_dict(self):
        # Step 1: Compute column sums
        col_sums = defaultdict(float)
        for (i, j), value in self.ahp_dict.items():
            col_sums[j] += value

        # Step 2: Normalize the values (build normalized AHP entries)
        normalized = defaultdict(float)
        for (i, j), value in self.ahp_dict.items():
            if col_sums[j] != 0:
                normalized[(i, j)] = value / col_sums[j]

        # Step 3: Compute row-wise mean to get the priority vector
        n = len(self.image_ids)
        row_sums = np.zeros(n)
        row_counts = np.zeros(n)

        for (i, j), value in normalized.items():
            row_sums[i] += value
            row_counts[i] += 1

        # Avoid division by zero
        priority_vector = np.array([
            row_sums[i] / row_counts[i] if row_counts[i] != 0 else 1.0 / n
            for i in range(n)
        ])
        
        self.priority_vector_dict = priority_vector

    ####################
    #  Matrix function #
    ####################
    
    def create_matrix_from_comparisons(self, df_):
        
        n = len(self.image_ids)
        self.votes_matrix = np.zeros((n, n, 2), dtype=int)

        for _, row in df_.iterrows():
            image_1, image_2, result = row["left_id"], row["right_id"], row["winner"]
            
            left = self.image_to_idx[row['left_id']]
            right = self.image_to_idx[row['right_id']]
            result = row['winner'].strip().lower()
            
            if result == 'left':
                self.votes_matrix[left][right][0] += 1
            elif result == 'right':
                self.votes_matrix[right][left][0] += 1
            elif result == 'equal':
                self.votes_matrix[left][right][1] += 1
                self.votes_matrix[right][left][1] += 1
    
    def build_sparse_ahp_from_matrix(self):
        n = len(self.image_ids)
        ahp_matrix = np.ones((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                num = self.votes_matrix[i, j]
                den = self.votes_matrix[j, i]
                numerator = num[0] + 0.5 * num[1]
                denominator = den[0] + 0.5 * num[1]
                ahp_matrix[i][j] = numerator / denominator if denominator != 0 else 1e6

        self.ahp_matrix = ahp_matrix

    def calculate_priority_vector_from_matrix(self):
        col_sums = self.ahp_matrix.sum(axis=0)
        normalized = self.ahp_matrix / col_sums
        priority_vector = normalized.mean(axis=1)
        
        self.priority_vector_matrix = priority_vector
        
