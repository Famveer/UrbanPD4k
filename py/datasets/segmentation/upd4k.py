import pandas as pd
import json
import ast
from typing import Literal
from .dataset import SegmentationDataset
from .ade20k import ADE20KDataset
from py.utils import verifyFile


class UPD4KDataset(SegmentationDataset):
    DATASET_LEN = 13
    
    ALLOWED_DATASETS = {"ade20k", "cityscapes"}
    
    GROUP_CLASSES = {
        "broken_damaged_bricks_wall": ["broken_damaged_bricks_wall"],
        "broken_damaged_pavement_road": ["broken_damaged_pavement_road"],
        "broken_window": ["broken_window"],
        "car_police": ["car_police"],
        "car_taxi": ["car_taxi"],
        "damaged_traffic_sign": ["damaged_traffic_sign"],
        "garbage_bag": ["garbage_bag"],
        "garbage_box": ["garbage_box"],
        "graffiti": ["graffiti"],
        "homeless": ["homeless"],
        "kiosk": ["kiosk"],
        "overhead_cable": ["overhead_cable"],
        "trashcan": ["trashcan"],
    }

    def normalize_dataset(self, name: str) -> str:
        name = name.lower()
        if name not in self.ALLOWED_DATASETS:
            raise ValueError(f"Dataset must be one of {ALLOWED_DATASETS}")
        return name

    def load(self, dataset: str = "upd4k", data_path: str = ""):
        if not verifyFile(f"{self.data_path}upd4k_categories.csv"):
            self.generate_upd4k_dataset()
        
        self._read_upd4k()
        
        if dataset.lower() != "upd4k":
            dataset_name = self.normalize_dataset(dataset)
            self.combine_datasets(dataset_name, data_path)
            self.dataset_used = f"{dataset_name}_upd4k"
        else:
            self.dataset_used = "upd4k"
        
        self.dataset_len = self.DATASET_LEN 
        self.labels      = self.data_df["main_class_name"].tolist()
 
        objects_df        = self.data_df.copy()
 
        self.CLASS_TO_GROUP = { cls: group for group, classes in self.GROUP_CLASSES.items()
                        for cls in classes
                     }
 
        self._finalise(objects_df)


    def _read_upd4k(self):
        df = pd.read_csv(f"{self.data_path}upd4k_categories.csv", sep=";", low_memory=False)
        
        df["class_id"]   = df["class_id"].astype(int)
        df["RGB_color"]  = df["RGB_color"].apply(ast.literal_eval)
        df["RGB_color_group"]  = df["RGB_color_group"].apply(ast.literal_eval)
        self.data_df   = df.copy()
        self.upd4k_labels = df["main_class_name"].tolist()
    

    def generate_upd4k_dataset(self):
        labelmap_path = f"{self.data_path}/labelmap.txt"
        
        data = []
        with open(labelmap_path, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("#") or not line:  # Skip comments and empty lines
                    continue
                parts = line.split(":")
                label = parts[0]
                rgb = tuple(map(int, parts[1].split(",")))  # Convert RGB values to a tuple
                hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)  # Convert to HEX
                isthing = parts[-1]
                data.append([label, str(rgb), hex_color, int(isthing)])  # Append label, RGB tuple, and HEX

        # Create DataFrame
        upd4k_df = pd.DataFrame(data, columns=["main_class_name", "RGB_color", "hex_color", "isthing"])
        upd4k_df = upd4k_df.iloc[1:, :].reset_index(drop=True).copy()
        upd4k_df["class_id"] = (upd4k_df.index).tolist()
        upd4k_df["class_name"] = upd4k_df["main_class_name"]
        
        upd4k_df['RGB_color'] = upd4k_df['RGB_color'].apply(ast.literal_eval)
        upd4k_df["group_class_name"] = upd4k_df["main_class_name"]
        upd4k_df["RGB_color_group"] = upd4k_df["RGB_color"]
        upd4k_df["hex_color_group"] = upd4k_df["hex_color"]
        
        upd4k_df.to_csv(f"{self.data_path}upd4k_categories.csv", sep=";", index=False)
        
        self.data_df = upd4k_df.copy()
        
    def combine_datasets(self, dataset: str = "", data_path: str = ""):
        if "ade20k" in dataset.lower():
            uss = ADE20KDataset(data_path=data_path)
        elif "cityscapes" in dataset.lower():
            #uss = CityScapesDataset(data_path=data_path)
            pass
        
        uss.load()
        new_objects_df = uss.get_urban_street_categories(by_groups=False)
        self.data_df["class_id"] = self.data_df["class_id"] + 1 + len(new_objects_df)
        
        new_objects_df = pd.concat([new_objects_df, self.data_df], ignore_index=True)
        self.data_df = new_objects_df.copy()
        
        self.DATASET_LEN += len(new_objects_df)
        
        self.GROUP_CLASSES = {**uss.GROUP_CLASSES, **self.GROUP_CLASSES}
    
