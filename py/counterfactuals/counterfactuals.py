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
        self.cfs_variations = pd.DataFrame()
        # number of changes
        self.cfs_num_variations = pd.DataFrame()
        # counterfactuals
        self.counterfactuals = pd.DataFrame()
        # Most closets counterfactuals
        self.nearest_cfs = pd.DataFrame()
        
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
                
                self._process_counterfactuals(dice_exp, sample, id_)
                
            except Exception as e:
                if verbose:
                    print(f"Error processing image {id_}: {e}")
    
    def _process_counterfactuals(self, dice_exp, sample, image_id):
        """Process and store counterfactual results."""
        # Extract counterfactual values
        cf_values = dice_exp.cf_examples_list[0].final_cfs_df.iloc[:, :-1].copy()
        
        # Calculate differences
        values_diff = cf_values.subtract(sample.iloc[0], axis=1)
        values_diff["image_id"] = image_id
        self.cfs_variations = pd.concat([self.cfs_variations, values_diff], ignore_index=True)
        
        # Count number of changes
        number_diff = pd.DataFrame(values_diff.ne(0).astype(int).sum()).T
        number_diff["image_id"] = image_id
        self.cfs_num_variations = pd.concat([self.cfs_num_variations, number_diff], ignore_index=True)
        
        # Find closest counterfactual
        self._find_closest_counterfactual(cf_values, sample, image_id)
        
        # Store counterfactual values
        cf_values["image_id"] = image_id
        self.counterfactuals = pd.concat([self.counterfactuals, cf_values], ignore_index=True)
    
    def _find_closest_counterfactual(self, cf_values, sample, image_id):
        """Find and process the closest counterfactual."""
        # Calculate Euclidean distances
        distances = euclidean_distances(cf_values, sample)
        closest_row_index = np.argmin(distances)
        closest_row = cf_values.iloc[closest_row_index]
        
        # Get predictions
        orig_predict = self.model.predict_proba(sample)
        new_predict = self.model.predict_proba(closest_row.to_frame().T)
        diff_predict = orig_predict - new_predict
        
        # Store results
        closest_row_diff = closest_row.subtract(sample.iloc[0]).to_frame().T
        closest_row_diff["image_id"] = image_id
        closest_row_diff["orig_prob"] = [orig_predict]
        closest_row_diff["new_prob"] = [new_predict]
        closest_row_diff["diff_prob"] = diff_predict[:, 0].tolist()[0]
        
        self.nearest_cfs = pd.concat([self.nearest_cfs, closest_row_diff], ignore_index=True)
    
    def plot_variations(self, fig_size=(40, 30), top_k=None, color_dict=None):
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
            
            features_df = self.cfs_num_variations.iloc[:, :-1]
            sorted_columns = features_df.median().sort_values(ascending=False).index.tolist()
            
            if top_k is not None:
                sorted_columns = sorted_columns[:top_k]
            
            sns_fig = sns.boxplot(
                data=features_df[sorted_columns],
                orient="h",
                flierprops=dict(marker='x', markersize=40, color='black', markeredgewidth=5),
                medianprops=dict(color="green", linewidth=10),
                notch=True,
                showcaps=False,
                color="steelblue",
                native_scale=True,
                palette=color_dict,
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
            self.cfs_variations = pd.read_csv(f"{load_path}/cfs_variations.csv", sep=";", low_memory=False)
            self.cfs_num_variations = pd.read_csv(f"{load_path}/cfs_num_variations.csv", sep=";", low_memory=False)
            self.counterfactuals = pd.read_csv(f"{load_path}/counterfactuals.csv", sep=";", low_memory=False)
            self.nearest_cfs = pd.read_csv(f"{load_path}/nearest_cfs.csv", sep=";", low_memory=False)
        
        except Exception as e:
            print("Error", e)
    
    
    def save(self, save_path):
        results = self.get_results()
        for df_name in results.keys():
            cur_df = results[df_name].copy()
            cur_df.to_csv(f"{save_path}/{df_name}.csv", sep=";", index=False)
    
    
    def get_results(self):
        """Return all result dataframes."""
        return {
            'cfs_variations': self.cfs_variations,
            'cfs_num_variations': self.cfs_num_variations,
            'counterfactuals': self.counterfactuals,
            'nearest_cfs': self.nearest_cfs
        }
