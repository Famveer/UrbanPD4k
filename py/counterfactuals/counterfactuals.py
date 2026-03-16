import re
import ast
import dice_ml
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class CounterfactualAnalyzer:
    def __init__(self, data_df=None, feature_names=None, model=None):
        """
        Initialize the CounterfactualAnalyzer.
        
        Parameters:
        -----------
        data_df : pd.DataFrame
            The input dataframe with predictions
        model : sklearn model
            Trained model for predictions
        feature_names : list
            List of feature column names
        """
        self.data_df = data_df
        self.model = model
        self.feature_names = feature_names
        
        # Result dataframes
        # original values - counterfactual
        self.features_variations = pd.DataFrame()
        # number of changes
        self.features_num_variations = pd.DataFrame()
        # counterfactuals
        self.counterfactuals = pd.DataFrame()
        # Most closets counterfactuals
        self.nearest_cf = pd.DataFrame()
        # Diff between otiginal and the most closets counterfactuals
        self.nearest_cf_variation = pd.DataFrame()
        
        # Initialize DiCE
        if data_df is not None and feature_names is not None and model is not None:
            self._setup_dice()
    
    def _setup_dice(self):
        """Set up DiCE ML components."""
        self.dice_data = dice_ml.Data(
            dataframe=self.data_df.iloc[:, 1:].copy(),
            continuous_features=self.feature_names,
            outcome_name='target'
        )
        
        self.dice_model = dice_ml.Model(self.model, backend="sklearn")
        self.exp = dice_ml.Dice(self.dice_data, self.dice_model)
    
    def generate_counterfactuals(self, 
                                  test_data_df,
                                  desired_target,
                                  total_cfs=10, 
                                  stopping_threshold=0.6,
                                  permitted_range=None,
                                  random_seed=42,
                                  features_to_vary="all",
                                  verbose=False,
                                  ):
        """
        Generate counterfactuals for all unsafe samples.
        
        Parameters:
        -----------
        total_cfs : int
            Number of counterfactuals to generate per sample
        stopping_threshold : float
            Minimal class probability of counterfactual
        permitted_range_max : float
            Maximum allowed value for features
        features_to_vary:
            list of features to vary e.g., ['tree', 'building', 'road']
        random_seed : int
            Random seed for reproducibility
        verbose : bool
            Whether to print progress
        """
        print(f"Analyzing {test_data_df.shape[0]} samples")
        for index, row in tqdm(test_data_df.iterrows()):
            sample = row[self.feature_names].to_frame().T
            id_ = row["image_id"]
            target_ = row["target"]
            
            try:
                dice_exp = self.exp.generate_counterfactuals(
                    sample,
                    total_CFs=total_cfs,
                    desired_class=desired_target,
                    verbose=verbose,
                    stopping_threshold=stopping_threshold, # min cf target class probability.
                    features_to_vary=features_to_vary,
                    #proximity_weight=1.0, # more close the counterfactuals are
                    #sparsity_weight=2.0,   # less features are changed
                    #diversity_weight=2.0, # more diverse the counterfactuals are.
                    permitted_range=permitted_range,
                    random_seed=random_seed
                )
                
                self._process_counterfactuals(dice_exp, 
                                              sample, 
                                              id_, 
                                              desired_target,
                                              min_probability=stopping_threshold,
                                              )
                
            except Exception as e:
                if verbose:
                    print(f"Error processing image {id_}: {e}")
    
    def _process_counterfactuals(self, 
                                 dice_exp, 
                                 sample, 
                                 image_id, 
                                 desired_target,
                                 min_probability=0.5,
                                 ):
        """Process and store counterfactual results."""
        # Extract counterfactual values
        cf_values = dice_exp.cf_examples_list[0].final_cfs_df.iloc[:, :-1].copy()
        
        # Calculate differences
        values_diff = cf_values.subtract(sample.iloc[0], axis=1)
        values_diff["image_id"] = image_id
        self.features_variations = pd.concat([self.features_variations, values_diff], ignore_index=True)
        
        # Count number of changes
        number_diff = pd.DataFrame(values_diff.ne(0).astype(int).sum()).T
        number_diff["image_id"] = image_id
        self.features_num_variations = pd.concat([self.features_num_variations, number_diff], ignore_index=True)
        
        # Find closest counterfactual that changes to desired class
        self._find_closest_counterfactual(cf_values, 
                                          sample, 
                                          image_id, 
                                          desired_target, 
                                          min_probability=min_probability,
                                          )
        
        # Store counterfactual values
        cf_values["image_id"] = image_id
        self.counterfactuals = pd.concat([self.counterfactuals, cf_values], ignore_index=True)
    
    
    def _find_closest_counterfactual(self, 
                                     cf_values, 
                                     sample, 
                                     image_id, 
                                     desired_target, 
                                     min_probability=0.5,
                                     ):
        """
        Find and process the closest counterfactual that changes prediction to desired class.
        
        Parameters:
        -----------
        cf_values : pd.DataFrame
            Generated counterfactual samples
        sample : pd.DataFrame
            Original sample
        image_id : str/int
            Image identifier
        desired_target : int
            Target class index (e.g., 0 for unsafe, 1 for safe)
        min_probability : float, optional (default=0.5)
            Minimum probability threshold for the desired class
        """
        # Get predictions for all counterfactuals
        cf_predictions = self.model.predict_proba(cf_values)
        
        # Get predicted class and probability for desired target
        cf_predicted_classes = np.argmax(cf_predictions, axis=1)
        cf_desired_class_probs = cf_predictions[:, desired_target]
        
        # Filter counterfactuals that:
        # 1. Predict the desired target class
        # 2. Have probability >= min_probability for the desired class
        valid_cf_mask = (cf_predicted_classes == desired_target) & (cf_desired_class_probs >= min_probability)
        valid_cf_indices = np.where(valid_cf_mask)[0]
        
        if len(valid_cf_indices) == 0:
            print(f"Warning: No counterfactuals found for image {image_id} that predict class {desired_target} with prob >= {min_probability}")
            # Fallback: try with any counterfactual that predicts desired class
            valid_cf_mask = cf_predicted_classes == desired_target
            valid_cf_indices = np.where(valid_cf_mask)[0]
            
            if len(valid_cf_indices) == 0:
                print(f"Error: No counterfactuals found for image {image_id} that predict class {desired_target}")
                return
        
        # Filter to only valid counterfactuals
        valid_cf_values = cf_values.iloc[valid_cf_indices]
        
        # Calculate Euclidean distances only for valid counterfactuals
        distances = euclidean_distances(valid_cf_values, sample)
        
        # Find the closest valid counterfactual
        closest_valid_index = np.argmin(distances)
        closest_row_index = valid_cf_indices[closest_valid_index]
        closest_row = cf_values.iloc[closest_row_index].copy()
        closest_row_T = closest_row.to_frame().T.copy()
        closest_row_T["image_id"] = image_id
        
        # Store the nearest counterfactual
        self.nearest_cf = pd.concat([self.nearest_cf, closest_row_T], ignore_index=True)
        
        # Get predictions
        orig_predict = self.model.predict_proba(sample)
        new_predict = self.model.predict_proba(closest_row.to_frame().T)
        diff_predict = new_predict - orig_predict
        
        # Store results with detailed information
        closest_row_diff = closest_row.subtract(sample.iloc[0]).to_frame().T
        closest_row_diff["image_id"] = image_id
        closest_row_diff["orig_prob"] = [orig_predict]
        closest_row_diff["new_prob"] = [new_predict]
        closest_row_diff["diff_prob"] = [diff_predict]
        closest_row_diff["orig_class"] = np.argmax(orig_predict, axis=1)[0]
        closest_row_diff["new_class"] = np.argmax(new_predict, axis=1)[0]
        closest_row_diff["desired_class_prob"] = cf_predictions[closest_row_index, desired_target]
        closest_row_diff["euclidean_dist"] = distances[closest_valid_index][0]
        
        self.nearest_cf_variation = pd.concat([self.nearest_cf_variation, closest_row_diff], ignore_index=True)
    
    def plot_variations(self, fig_size=(40, 30), palette=None, top_k=None):
        """
        Plot number of variations in counterfactuals.
        
        Parameters:
        -----------
        fig_size : tuple
            Figure size (width, height)
        top_k : int, optional
            Show only top k features
        color_dict : dict, optional
            Color dictionary for palette
        """
        try:
            fig, ax = plt.subplots(
                figsize=fig_size, 
                nrows=1, 
                ncols=1, 
                constrained_layout=True
            )
            
            features_df = self.features_num_variations.iloc[:, :-1]
            sorted_columns = features_df.median().sort_values(ascending=False).index.tolist()
            
            if top_k is not None:
                sorted_columns = sorted_columns[:top_k]
                
            if palette is not None:
                palette = {
                    k: tuple(v / 255 for v in rgb) if max(rgb) > 1 else rgb
                    for k, rgb in palette.items()
                }
            
            sns_fig = sns.boxplot(
                data=features_df[sorted_columns],
                orient="h",
                flierprops=dict(marker='x', markersize=40, color='black', markeredgewidth=5),
                medianprops=dict(color="green", linewidth=10),
                notch=True,
                showcaps=False,
                color="steelblue",
                native_scale=True,
                palette=palette,
            )
            
            sns_fig.set_title("CounterFactuals", fontsize=0)
            sns_fig.set_ylabel("", fontsize=0)
            sns_fig.set_xlabel('Number of variations', fontsize=70)
            sns_fig.tick_params(axis='x', rotation=0, labelsize=55)
            sns_fig.tick_params(axis='y', labelsize=80)
            
            xticks_ = list(range(0, features_df.values.max() + 4, 5))
            sns_fig.set_xticks(xticks_)
            sns_fig.set_xticklabels([str(i) for i in xticks_])
            
            sns_fig.margins(x=0.1)
            ax.grid(True)
            
            plt.show()
            
        except Exception as e:
            print(f"Error plotting: {e}")
    
    
    def load(self, load_path):
        try:
            self.counterfactuals = pd.read_csv(f"{load_path}/counterfactuals.csv", sep=";", low_memory=False)
        except Exception as e:
            print("Error", e)
            
        try:
            self.features_variations = pd.read_csv(f"{load_path}/features_variations.csv", sep=";", low_memory=False)
        except Exception as e:
            print("Error", e)
        
        try:
            self.features_num_variations = pd.read_csv(f"{load_path}/features_num_variations.csv", sep=";", low_memory=False)
        except Exception as e:
            print("Error", e)
        
        try:    
            self.nearest_cf = pd.read_csv(f"{load_path}/nearest_cf.csv", sep=";", low_memory=False)
        except Exception as e:
            print("Error", e)
        
        try:
            cur_df = pd.read_csv(f"{load_path}/nearest_cf_variation.csv", sep=";", low_memory=False)
            cur_df["orig_prob"] = cur_df["orig_prob"].apply(ast.literal_eval)
            cur_df["new_prob"] = cur_df["new_prob"].apply(ast.literal_eval)
            cur_df["diff_prob"] = cur_df["diff_prob"].apply(ast.literal_eval)
            self.nearest_cf_variation = cur_df.copy()
        except Exception as e:
            print("Error", e)
        
    
    def save(self, save_path):
        results = self.get_results()
        for df_name in results.keys():
            cur_df = results[df_name].copy()
            if df_name == "nearest_cf_variation":
                cur_df["orig_prob"] = cur_df["orig_prob"].apply(lambda x: x.tolist())
                cur_df["new_prob"] = cur_df["new_prob"].apply(lambda x: x.tolist())
                cur_df["diff_prob"] = cur_df["diff_prob"].apply(lambda x: x.tolist())
            cur_df.to_csv(f"{save_path}/{df_name}.csv", sep=";", index=False)
    
    
    def get_results(self):
        """Return all result dataframes."""
        return {
            'counterfactuals': self.counterfactuals,
            'nearest_cf': self.nearest_cf,
            'nearest_cf_variation': self.nearest_cf_variation,
            'features_variations': self.features_variations,
            'features_num_variations': self.features_num_variations,
        }
