from joblib import Parallel, delayed
import pandas as pd
from multiprocessing import Pool
import numpy as np
from collections import defaultdict

class EloRatings:
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

    ################################
    #  Elo Rating System functions #
    ################################
    
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
        
        self.images_df = images_df

    def expected_score(self, rating_1, rating_2):
        return 1 / (1 + 10 ** ( (rating_2 - rating_1) / 400 ) )

    def adaptive_K(self, rating1, rating2, match_count1, match_count2):
        """Adjust K-factor based on rating difference and number of matches."""
        rating_diff = abs(rating1 - rating2)
        
        # Adjust K based on rating difference
        K = max(self.min_K, min(self.max_K, self.initial_K - rating_diff / 100))
        
        # If either player has played fewer matches, increase K
        if match_count1 < 10 or match_count2 < 10:  # Consider a higher K if the entity is new
            K = max(K, self.initial_K)  # Cap at initial_K for new players
        return K

    def calculate_elo_ratings(self, metric="safety", initial_rating=0, K=100, max_K=40, min_K=10, adaptative_K=False):
    
        self.prepare_matches(metric=metric)
        df_ = self.get_matches().copy()
        print("Analyzing", df_.shape[0], metric, "comparisons")
        
        self.initial_K = K
        self.max_K = max_K
        self.min_K = min_K
        
        self.current_ratings = {}
        self.match_count = {}

        for index, row in df_.iterrows():
            image_1, image_2, result = row["left_id"], row["right_id"], row["winner"]
            
            current_rating_1 = self.current_ratings.get( image_1, initial_rating )
            current_rating_2 = self.current_ratings.get( image_2, initial_rating )
            match_count_1 = self.match_count.get( image_1, 0 )
            match_count_2 = self.match_count.get( image_2, 0 )

            # Calculate expected scores
            expected_rating_1 = self.expected_score(current_rating_1, current_rating_2)
            expected_rating_2 = 1 - expected_rating_1

            # Actual scores based on the result
            # Actual scores based on the result
            if result == "equal":  # Draw
                score_1, score_2 = 0.5, 0.5
            elif result == "left": # Image1 wins
                score_1, score_2 = 1, 0
            else:                  # Image2 wins
                score_1, score_2 = 0, 1
            
            # Adjust K based on rating difference and match counts
            if adaptative_K:
                K = self.adaptive_K(current_rating_1, current_rating_2, match_count_1, match_count_2)

            # Update ratings
            self.current_ratings[image_1] = current_rating_1 + K * (score_1 - expected_rating_1)
            self.current_ratings[image_2] = current_rating_2 + K * (score_2 - expected_rating_2)
            
            # Update match count
            self.match_count[image_1] = match_count_1 + 1
            self.match_count[image_2] = match_count_2 + 1
    
    def normalize_elo_ratings(self, normalize=True, min_range=0, max_range=10):
        ratings_df = pd.DataFrame(list(self.current_ratings.items()), columns=["image_id", "EloRating"])
        
        if normalize:
            # Normalize Elo ratings to the range [0, 10]
            min_rating = ratings_df["EloRating"].min()
            max_rating = ratings_df["EloRating"].max()
            
            ratings_df["EloScore"] = min_range + ((ratings_df["EloRating"] - min_rating) / (max_rating - min_rating)) * (max_range - min_range)
            
        ratings_df = pd.merge(ratings_df, self.images_df, on="image_id", how="left")
        self.ratings_df = ratings_df
    
    def get_elo_ratings(self):
        return self.ratings_df
    
