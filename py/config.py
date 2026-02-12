import os
from pydantic_settings import BaseSettings
import ast
from dotenv import load_dotenv
load_dotenv()

class Config(BaseSettings):

    DATA_PATH: str = os.getenv('DATA_PATH')
    MODEL_PATH: str = os.getenv('MODEL_PATH')
    
    ML_TASK: str = os.getenv('ML_TASK', "classifications")
    MODEL_TASK_NAME: str = os.getenv('MODEL_TASK_NAME', "RandomForest")
    PRECEPTION_METRIC: str = os.getenv('PRECEPTION_METRIC', "safety")

    SEG_MODEL_NAME: str = os.getenv('SEG_MODEL_NAME', "OneFormer_Swin_Large")
    SEG_DATASET: str = os.getenv('SEG_DATASET', "ade20k") # cityscapes
    CITY_STUDIED: str = os.getenv('CITY_STUDIED', "Rio De Janeiro")

    USE_UPD: bool = ast.literal_eval(os.getenv('USE_UPD', "True"))
    FILTER_FEATURES: bool = ast.literal_eval(os.getenv('FILTER_FEATURES', "False"))
    BY_GROUPS: bool = ast.literal_eval(os.getenv('BY_GROUPS', "True"))
    BINARIZE_FEATURES: bool  = ast.literal_eval(os.getenv('BINARIZE_FEATURES', "False"))
    
    TOP_K_FEATURES: int = int(os.getenv('TOP_K_FEATURES', "15"))
    NUMBER_CFS: int = int(os.getenv('NUMBER_CFS', "50"))
    
    CHANGE_THRESHOLD: float = float(os.getenv("CHANGE_THRESHOLD", "0.001"))
    
    RANDOM_STATE: int = int(os.getenv('RANDOM_STATE', "42"))
