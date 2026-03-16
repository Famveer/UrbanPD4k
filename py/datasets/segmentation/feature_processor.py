import ast
import json
import pandas as pd
import numpy as np
from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt

class FeatureProcessor:

    def __init__(self, columns_list=["image_id", "seg_image_path", "seg_overlay_image_path", "mask_path", "ratio_path"],
                ):
        self.columns_list = columns_list
        

    def process(self, data_df,
                      class_to_group,
                      threshold=0.05,
                      filter_features=False,  
                      aggregate_classes=False,
                      binarize_values=False):
        
        input_df = data_df.copy()
        
        if filter_features:
            input_df = self.filter_features(input_df, threshold=threshold)

        if aggregate_classes:
            input_df = self.aggregate_classes_by_groups(input_df, class_to_group)
            
        if binarize_values:
            input_df = self.binarize_vectors(input_df)
        
        return input_df
    

    def filter_features(self, input_df, threshold=0.05):
        _df = input_df.copy()
        _df.drop(columns=self.columns_list, errors='ignore', inplace=True)
        
        _df[_df > 0] = 1
        _df = (_df.sum()/len(_df) ).reset_index()
        _df.sort_values(by=0, ascending=False, inplace=True)
        
        if 0>=threshold:
            df_grouped = _df.copy()
            features_to_keep = df_grouped["index"].tolist()
        
        elif 1>threshold:
            df_grouped = _df[_df[0]>= threshold].copy()
            features_to_keep = df_grouped["index"].tolist()
        
        else:
            df_grouped = _df[:threshold].copy()
            features_to_keep = df_grouped["index"].tolist()
        
        r_df= input_df[self.columns_list+features_to_keep].copy()
        return r_df
    

    def aggregate_classes_by_groups(self, input_df, class_to_group):
        df_to_group = input_df.copy()
        df_to_group.drop(columns=self.columns_list, errors='ignore', inplace=True)
        
        mapped_column_names = [class_to_group.get(col, col) for col in df_to_group.columns]
        
        df_to_group.columns = mapped_column_names
        
        # deprecated pandas <=2.1.0
        # df_grouped = df_to_group.groupby(df_to_group.columns, axis=1).sum()
        df_grouped = df_to_group.T.groupby(df_to_group.columns).sum().T
        
        grouped_df= pd.concat([input_df[self.columns_list].copy(), df_grouped], axis=1)
        return grouped_df
    

    def binarize_vectors(self, input_df):
        features_df = input_df.copy()
        features_df.drop(columns=self.columns_list, errors='ignore', inplace=True)
        features_df = (features_df > 0).astype(int)
        
        binarize_df = pd.concat([input_df[self.columns_list].copy(), features_df], axis=1)
        return binarize_df
    

    def calculate_features_presence(self, df):
        features_df = df.copy()
        features_df[features_df > 0] = 1
        features_df = (features_df.sum()/len(features_df) ).reset_index()
        features_df.sort_values(by=0, ascending=False, inplace=True)

        return features_df
        
    # PLOT
    def plot_presences(self, features_df, palette=None, fig_size=(40,20), top_k=None):
        try:
            fig, ax = plt.subplots(figsize=fig_size, nrows=1, ncols=1, sharex=False, sharey=False)
            
            if top_k is not None:
                features_df = features_df.head(top_k).copy()
            
            if palette is not None:
                palette = {
                    k: tuple(v / 255 for v in rgb) if max(rgb) > 1 else rgb
                    for k, rgb in palette.items()
                }
            
            sns_fig = sns.barplot(
                        data=features_df,
                        x=features_df.columns[1],
                        y=features_df.columns[0],
                        ax=ax,
                        palette=palette, 
                        # order=estado_df.sort_values(estado_df.columns[1], ascending=False)[estado_df.columns[0]]
                       )

            sns_fig.set_title(f"Elements presence", fontsize=40)
            sns_fig.set_ylabel(f"Number of comparisons", fontsize=0)
            sns_fig.set_xlabel('% of images containing this object', fontsize=40)

            # rotate the axis ticklabels
            _ = sns_fig.tick_params(axis='x', rotation=0, labelsize=45)
            
            # rotate the axis ticklabels
            _ = sns_fig.tick_params(axis='y', labelsize=40)

            # add annotation
            #_ = sns_fig.bar_label(sns_fig.containers[0], fmt='%0.0f', fontsize=15,rotation=0)

            # add a space on y for the annotations
            #sns_fig.margins(x=0.1)
            ax.grid(True)

            plt.show()
        except Exception as e:
            print(e)
    
    
    def plot_boxplots(self, features_df, palette=None, fig_size=(40,20), top_k=None, pixel_presence=True):
        try:
            fig, ax = plt.subplots(figsize=fig_size, nrows=1, ncols=1, sharex=False, sharey=False, constrained_layout=True)
            
            sorted_columns = features_df.median().sort_values(ascending=False).index.tolist()
            
            if top_k is not None:
                sorted_columns = sorted_columns[:top_k].copy()
            
            if pixel_presence:
                features_df[sorted_columns] = features_df[sorted_columns]/100
            
            if palette is not None:
                palette = {
                    k: tuple(v / 255 for v in rgb) if max(rgb) > 1 else rgb
                    for k, rgb in palette.items()
                }
        
            sns_fig = sns.boxplot(
                        data=features_df[sorted_columns],
                        orient="h",
                        # dodge=True,
                        flierprops=dict(marker='o', markersize=30),
                        # showfliers=True, 
                        medianprops=dict(color="green", linewidth=5),
                
                        notch=True, showcaps=False,
                        color="steelblue",
                        native_scale=True,
                        palette=palette,
                        # order=estado_df.sort_values(estado_df.columns[1], ascending=False)[estado_df.columns[0]]
                       )
            
            sns_fig.set_title(f"Object presence", fontsize=50)
            sns_fig.set_ylabel(f"", fontsize=0)
            sns_fig.set_xlabel('% of pixels within images', fontsize=50)
            
            # rotate the axis ticklabels
            _ = sns_fig.tick_params(axis='x', rotation=0, labelsize=45)
            
            # rotate the axis ticklabels
            _ = sns_fig.tick_params(axis='y', labelsize=40)
            
            #sns_fig.set_xticks(list(range(10)))
            #sns_fig.set_xticklabels([str(i) for i in list(range(10))])
            
            # add a space on y for the annotations
            sns_fig.margins(x=0.1)
            ax.grid(True)
            
            plt.show()
        except Exception as e:
            print(e)

