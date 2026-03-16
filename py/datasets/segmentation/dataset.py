import ast
import json
import pandas as pd
import numpy as np

class SegmentationDataset:
    _GROUP_COLORS = {
        "air_vehicle":             ("(0, 255, 82)",    "#00FF52"),
        "animal":                  ("(255, 0, 122)",   "#FF007A"),
        "body_water":              ("(61, 230, 250)",  "#3DE6FA"),
        "city_elements":           ("(0, 71, 255)",    "#0047FF"),
        "clothes_object":          ("(0, 112, 255)",   "#0070FF"),
        "construction":            ("(180, 120, 120)", "#B47878"),
        "floor":                   ("(140, 140, 140)", "#8C8C8C"),
        "human":                   ("(150, 5, 61)",    "#96053D"),
        "indoor_object":           ("(204, 5, 255)",   "#CC05FF"),
        "miscellaneous":           ("(0, 61, 255)",    "#003DFF"),
        "nature_object":           ("(143, 255, 140)", "#8FFF8C"),
        "outdoor_object":          ("(230, 230, 230)", "#E6E6E6"),
        "sea_vehicle":             ("(173, 255, 0)",   "#ADFF00"),
        "sea_vehicle_related":     ("(71, 0, 255)",    "#4700FF"),
        "sky":                     ("(6, 230, 230)",   "#06E6E6"),
        "terrain_vehicle":         ("(255, 0, 20)",    "#FF0014"),
        "terrain_vehicle_related": ("(255, 214, 0)",   "#FFD600"),
        "vegetation":              ("(4, 200, 3)",     "#04C803"),
    }

 
    def __init__(self, data_path: str): 
        self.data_path        = data_path
        self.objects_df       = None
        self.groups_df       = None
        self.dataset_used     = None
        self.dataset_len      = 0
        self.labels           = []
 
    # ------------------------------------------------------------------
    # Interface – subclasses must implement
    # ------------------------------------------------------------------
 
    def load(self, with_upd4k: bool = False):
        """Load class metadata. Must set objects_df, color_dict, etc."""
        raise NotImplementedError
 
    def get_group_classes(self) -> dict:
        if not self.GROUP_CLASSES:
            raise RuntimeError("Call load() before get_groups().")
        return self.GROUP_CLASSES
   
    def get_class_groups(self) -> dict:
        if not self.CLASS_TO_GROUP:
            raise RuntimeError("Call load() before get_groups().")
        return self.CLASS_TO_GROUP
    
    def get_color_dict(self, by_name=False, by_group=False):
        key_column = "class_"
        value_column = "RGB_color"
        
        if by_group:
            key_column = "group_" + key_column
            value_column += "_group"
        else:
            key_column = "main_" + key_column
        
        if by_name:
            key_column += "name"
        else:
            key_column += "id"
         
        color_dict = self.objects_df.set_index(key_column)[value_column].to_dict()
        return color_dict
    
    # ------------------------------------------------------------------
    # Shared metadata accessors
    # ------------------------------------------------------------------
 
    def get_urban_street_categories(self, by_groups: bool = False) -> pd.DataFrame:
        """
        Return class metadata.
 
        Parameters
        ----------
        by_groups : bool
            True  → one row per group (aggregated, with colour corrections).
            False → full per-class objects_df.
        """
        if by_groups:
            return self.groups_df 
        return self.objects_df
 
    def _street_categories_group(self, objects_df) -> pd.DataFrame:
    
        class_ids  = objects_df.copy().groupby("group_class_name")["class_id"].agg(list).reset_index()
        group_df     = pd.merge( objects_df[["group_class_name", "RGB_color_group", "hex_color_group"]].drop_duplicates(), class_ids, on="group_class_name" )
        group_df.rename(columns={"class_id": "classes"}, inplace=True)
        group_df["num_classes"] = group_df["classes"].apply(len)
        group_df["group_class_id"] = range(1, len(group_df) + 1)
        group_df = group_df[['group_class_id', 'group_class_name', 'RGB_color_group', 'hex_color_group', 'classes', 'num_classes']].copy()
        return group_df
        
    def get_vectorized_map(self, group_df):
    
        class_to_group_by_id = {cls: row['group_class_id']
                                     for _, row in group_df.iterrows()
                                     for cls in row['classes']
                                  }
    
        vectorized_map = np.vectorize(lambda x: class_to_group_by_id.get(x, x))
        return vectorized_map
 
    # ------------------------------------------------------------------
    # Internal helper: finalise shared state after subclass load
    # ------------------------------------------------------------------
 
    def _finalise(self, objects_df: pd.DataFrame):
        """Store objects_df and build color_dict after a subclass load."""
        self.groups_df = self._street_categories_group(objects_df)
        
        group_name_id = dict(zip(self.groups_df["group_class_name"].tolist(), self.groups_df["group_class_id"].tolist()))
        
        objects_df["group_class_id"] = objects_df["group_class_name"].map(group_name_id)
        
        self.objects_df = objects_df
        
