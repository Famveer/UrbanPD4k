import pandas as pd
import json
import ast
from .dataset import SegmentationDataset
from py.utils import verifyFile

class ADE20KDataset(SegmentationDataset):
    DATASET_LEN = 150
 
    # Complete group → class mapping derived from ade20k_categories.csv
    GROUP_CLASSES = {
        "air_vehicle":             ["airplane"],
        "animal":                  ["animal"],
        "body_water":              ["water", "sea", "river", "waterfall", "lake"],
        "city_elements":           ["signboard", "streetlight", "pole", "stoplight"],
        "clothes_object":          ["clothes", "bag"],
        "construction":            ["wall", "building", "ceiling", "door", "house", "fence",
                                    "column", "skyscraper", "bridge", "bar", "shack",
                                    "tower", "stadium", "fountain"],
        "floor":                   ["floor", "road", "sidewalk", "ground", "path", "land"],
        "human":                   ["person"],
        "indoor_object":           ["bed", "cabinet", "table", "curtain", "chair", "painting",
                                    "sofa", "shelf", "mirror", "carpet", "armchair", "seat",
                                    "desk", "closet", "lamp", "bathtub", "railing", "sofa_pillow",
                                    "pedestal", "box", "dresser", "counter", "sink", "fireplace",
                                    "refrigerator", "stairs", "vitrine", "billiard", "pillow",
                                    "bookcase", "coffee", "toilet", "book", "countertop", "stove",
                                    "kitchen", "computer", "swivel_chair", "arcade", "towel",
                                    "light", "chandelier", "stand", "television", "handrail",
                                    "escalator", "puff", "bottle", "sideboard", "conveyor",
                                    "canopy", "washer", "plaything", "stool_chair", "barrel",
                                    "cradle", "oven", "ball", "tank", "microwave", "dishwasher",
                                    "screen_projection", "blanket", "hood", "sconce", "vase",
                                    "tray", "ashcan", "fan", "crt_screen", "plate", "monitor",
                                    "shower"],
        "miscellaneous":           ["blind", "stage", "basket", "food", "trade", "flowerpot",
                                    "sculpture", "bulletin", "glass", "clock", "flag"],
        "nature_object":           ["mountain", "rock", "sand", "hill"],
        "outdoor_object":          ["windowpane", "grandstand", "runway", "screen_door",
                                    "stairway", "bench", "awning", "poster",
                                    "swimming_pool", "tent", "steps"],
        "sea_vehicle":             ["boat", "ship"],
        "sea_vehicle_related":     ["pier"],
        "sky":                     ["sky"],
        "terrain_vehicle":         ["car", "bus", "truck", "van", "motorcycle", "bicycle"],
        "terrain_vehicle_related": ["radiator"],
        "vegetation":              ["tree", "grass", "plant", "field", "flower", "palm"],
    }
    
    CLASS_TO_GROUP = { cls: group for group, classes in GROUP_CLASSES.items()
                        for cls in classes
                     }
 
    def load(self):
        if not verifyFile(f"{self.data_path}ade20k_categories.csv"):
            self.generate_ade20k_dataset()
        
        self._read_ade20k()
 
        self.dataset_len = self.DATASET_LEN
        self.labels      = self.data_df["main_class_name"].tolist()
 
        objects_df        = self.data_df.copy()
        self.dataset_used = "ade20k"
 
        self._finalise(objects_df)
 
    # ------------------------------------------------------------------
    def _read_ade20k(self):
        """Load the ADE20K category CSV and group colour table."""
        # Group colour palette
        df = pd.read_csv(f"{self.data_path}ade20k_categories.csv", sep=";", low_memory=False)
        
        df["class_id"]   = df["class_id"].astype(int)
        df["RGB_color"]  = df["RGB_color"].apply(ast.literal_eval)
        df["RGB_color_group"]  = df["RGB_color_group"].apply(ast.literal_eval)
        self.data_df   = df.copy()
        
    def read_ade20k_isthings(self):
        """Load ADE20K panoptic (is_thing) JSON."""
        with open(f"{self.data_path}ade20k_panoptic.json", "r") as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict(data, orient="index").reset_index()
        df.rename(columns={"index": "class_id"}, inplace=True)
        df.drop(columns=["name"], inplace=True)
        df["class_id"]          = df["class_id"].apply(lambda x: int(x) + 1)
        self.ade20k_panoptic_df = df.copy()
 
    def generate_ade20k_dataset(self):
        """
        Build the ADE20K class CSV from the raw ODS colour-coding file
        and the panoptic JSON. Writes result to disk and sets self.data_df.
        """
        self.read_ade20k_isthings()
 
        df = pd.read_excel(f"{self.data_path}color_coding_classes.ods", engine="odf")
        df["main_class_name"] = df["class_name"].apply(lambda x: x.split(";")[0])
        df.loc[df["class_id"] == 59,  "main_class_name"] = "screen_door"
        df.loc[df["class_id"] == 131, "main_class_name"] = "screen_projection"
        df.loc[df["class_id"] == 49,  "RGB_color"]  = "(140, 140, 200)"
        df.loc[df["class_id"] == 49,  "hex_color"]  = "#8C8CC8"
        
        df.loc[df["class_id"] == 14,  "main_class_name"] = "ground"
        df.loc[df["class_id"] == 29,  "main_class_name"] = "carpet"
        df.loc[df["class_id"] == 36,  "main_class_name"] = "closet"
        df.loc[df["class_id"] == 40,  "main_class_name"] = "sofa_pillow"
        df.loc[df["class_id"] == 41,  "main_class_name"] = "pedestal"
        df.loc[df["class_id"] == 45,  "main_class_name"] = "dresser"
        df.loc[df["class_id"] == 56,  "main_class_name"] = "vitrine"
        df.loc[df["class_id"] == 57,  "main_class_name"] = "billiard"
        df.loc[df["class_id"] == 76,  "main_class_name"] = "swivel_chair"
        df.loc[df["class_id"] == 80,  "main_class_name"] = "shack"
        df.loc[df["class_id"] == 89,  "main_class_name"] = "stand"
        df.loc[df["class_id"] == 92,  "main_class_name"] = "stadium"
        df.loc[df["class_id"] == 93,  "main_class_name"] = "clothes"
        df.loc[df["class_id"] == 96,  "main_class_name"] = "handrail"
        df.loc[df["class_id"] == 98,  "main_class_name"] = "puff"
        df.loc[df["class_id"] == 100,  "main_class_name"] = "sideboard"
        df.loc[df["class_id"] == 106,  "main_class_name"] = "conveyor"
        df.loc[df["class_id"] == 110,  "main_class_name"] = "swimming_pool"
        df.loc[df["class_id"] == 111,  "main_class_name"] = "stool_chair"
        df.loc[df["class_id"] == 117,  "main_class_name"] = "motorcycle"
        df.loc[df["class_id"] == 122,  "main_class_name"] = "steps"
        df.loc[df["class_id"] == 126,  "main_class_name"] = "flowerpot"
        df.loc[df["class_id"] == 137,  "main_class_name"] = "stoplight"
        df.loc[df["class_id"] == 142,  "main_class_name"] = "crt_screen"
        
        
        df = df[["class_id", "main_class_name", "class_name", "RGB_color", "hex_color"]].copy()
        df  = pd.merge(df, self.ade20k_panoptic_df, on="class_id", how="left")
        
        df["group_class_name"] = df["main_class_name"].map(self.CLASS_TO_GROUP)
        df["RGB_color_group"] = df["group_class_name"].map(lambda g: self._GROUP_COLORS.get(g, (None, None))[0])
        df["hex_color_group"] = df["group_class_name"].map(lambda g: self._GROUP_COLORS.get(g, (None, None))[1])
        
        df.to_csv(f"{self.data_path}ade20k_categories.csv", sep=";", index=False)
        
