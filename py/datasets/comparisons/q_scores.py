from joblib import Parallel, delayed
import pandas as pd
from multiprocessing import Pool
import numpy as np

class QScores:
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
    #  QScores functions #
    ######################
    
    def filter_player(self, cat_df, player="left", against="right"):
        df_ = cat_df[[f"{player}_id", "winner", "category", f"{player}_lat", f"{player}_long", f"{player}_city", f"{player}_country", f"{player}_continent", f"{against}_id"]].copy()
        df_.rename(columns={f"{player}_id": "image_id", f"{player}_lat": "lat", f"{player}_long":"long", f"{player}_city":"city", f"{player}_country":"country", f"{player}_continent":"continent", f"{against}_id": "against"}, inplace=True)
        df_["match"] = df_["winner"].apply(lambda x: self.map_round(x, player=player))
        
        df_.drop(columns=["winner"], inplace=True)
        df_.sort_values(by=["image_id", "category"], inplace=True)
        
        return df_
    
    def prepare_matches(self, metric="safety"):
    
        df_ = self.comparisons_df[self.comparisons_df["category"]==metric].copy()
    
        left_df = self.filter_player(df_, player="left", against="right")
        right_df = self.filter_player(df_, player="right", against="left")
        
        matches_df = pd.concat([left_df, right_df], ignore_index=True)
        matches_df.sort_values(by=["image_id", "category"], inplace=True)
        '''
        matches_df['draws'] = matches_df['match'].apply(lambda x: 1 if x == 'draw' else 0)
        matches_df['loses'] = matches_df['match'].apply(lambda x: 1 if x == 'lose' else 0)
        matches_df['wins'] = matches_df['match'].apply(lambda x: 1 if x == 'win' else 0)
        self.matches_df_ = matches_df.groupby(['image_id', 'lat', 'long', 'city', 'country', 'continent', 'category']).sum().reset_index()
        '''
        
        match_df = pd.pivot_table(matches_df,
                                 index=['image_id', 'lat', 'long', 'city', 'country', 'continent', 'category'],
                                 columns=["match"],
                                 values=["against"],
                                 aggfunc={
                                     "against": list
                                 }).reset_index()
        match_df.columns = ["_".join([col[1], col[0]]) if len(col[1])>=1 and len(col[0])>=1 else "".join(col) for col in match_df.columns]
        
        match_df['win_against'] = match_df['win_against'].apply(lambda x: x if isinstance(x, list) else [])
        match_df['draw_against'] = match_df['draw_against'].apply(lambda x: x if isinstance(x, list) else [])
        match_df['lose_against'] = match_df['lose_against'].apply(lambda x: x if isinstance(x, list) else [])

        self.matches_df = match_df
    
    def calculate_ponderates(self, row, df):
        lose_sum = df[df["image_id"].isin(row["lose_against"])]["lose_rate"].sum()
        win_sum = df[df["image_id"].isin(row["win_against"])]["win_rate"].sum()
        draw_sum = df[df["image_id"].isin(row["draw_against"])]["draw_rate"].sum()
        return lose_sum, win_sum, draw_sum
    
    def calculate_q_scores(self, metric="safety"):
        
        self.prepare_matches(metric=metric)
        df_ = self.get_matches().copy()
        print("Analyzing", df_.shape[0], metric, "images")
        
        df_['draws'] = df_['draw_against'].apply(len)
        df_['loses'] = df_['lose_against'].apply(len)
        df_['wins'] = df_['win_against'].apply(len)
        
        df_["total_games"] = df_["wins"] + df_["draws"] + df_["loses"]
        df_["win_rate"] = df_["wins"]/df_["total_games"]
        df_["lose_rate"] = df_["loses"]/df_["total_games"]
        df_["draw_rate"] = df_["draws"]/df_["total_games"]
        
        df_["win_weight"] = df_["win_against"].apply(lambda x: df_[df_["image_id"].isin(x)]["win_rate"].sum())
        df_["draw_weight"] = df_["draw_against"].apply(lambda x: df_[df_["image_id"].isin(x)]["draw_rate"].sum())
        df_["lose_weight"] = df_["lose_against"].apply(lambda x: df_[df_["image_id"].isin(x)]["lose_rate"].sum())
        
        # Q = 10/3*(W + w - l + 1)
        
        df_["W"] = df_.apply(lambda row: row["win_weight"]/row["wins"] if row["wins"]>0 else 0, axis=1)#, result_type="expand")
        df_["D"] = df_.apply(lambda row: row["draw_weight"]/row["draws"] if row["draws"]>0 else 0, axis=1)
        df_["L"] = df_.apply(lambda row: row["lose_weight"]/row["loses"] if row["loses"]>0 else 0, axis=1)

        df_["Qscore"] = 10/3*(df_["win_rate"] + df_["W"] - df_["L"] + 1)
        
        self.q_scores = df_
        
    def get_q_scores(self):
        return self.q_scores
    
    ####################
    #  Maping function #
    ####################
        
    def map_round(self, choice, player="left"):
        if choice=="equal":
            return "draw"
        elif choice==player:
            return "win"
        else:
            return "lose"

    def map_perception(self, metric):
        if metricin in ["safety", "lively", "wealthy", "beautiful"]:
            return "positive"
        elif metric in ["depressing", "boring"]:
            return "negative"
        else:
            return "none"
    
