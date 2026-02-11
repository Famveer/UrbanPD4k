from py.models.basemodel import BaseModel

from .oneformer_swin import OneFormer_Swin
from .mask2former_swin import Mask2Former_Swin
from .segformer_b import SegFormer_B
from .psp_resnet50_dilated import PSP_ResNet50_Dilated

import numpy as np
from PIL import Image
from collections import Counter

class ImageSegmenter(BaseModel):
    def __init__(self, 
                 model_name=None, 
                 dataset="ade20k", 
                 device=None, 
                 model_path=None,
                 ):
                 
        self.dataset = dataset
        
        if model_path is not None:
            self.model_path = model_path
        
        super().__init__(model_name, device)
    
    def model_zoo(self):
        model_zoo = [
                     "OneFormer_Swin_Large", "OneFormer_Swin_Tiny", "OneFormer_Dinat_Large", 
                     "Mask2Former_Swin_Large", "Mask2Former_Swin_Base", "Mask2Former_Swin_Small", "Mask2Former_Swin_Tiny", 
                     "SegFormer_B0", "SegFormer_B1", "SegFormer_B2", "SegFormer_B3", "SegFormer_B4", "SegFormer_B5", 
                    ]
    
        print("Model zoo ADE20K and CityScapes:", model_zoo, "\n" )
        print("Model Zoo only ADE20K-pytorch: PSP_ResNet50_Dilated \n")
        print("Model Zoo only ADE20K-keras: DeepLabV3_Exception65 \n")
        
    def initialize_model(self, model_name):
    
        if model_name is not None:
            if "oneformer" in model_name.lower():
                arch_model = "_".join(model_name.split("_")[-2:])
                self.model = OneFormer_Swin(arch_model=arch_model,  
                                            dataset=self.dataset,
                                            device=self.device,
                                            )
                
            elif "segformer_b" in model_name.lower():
                arch_model = model_name.split("_")[-1]
                self.model = SegFormer_B(arch_model=arch_model, 
                                         dataset=self.dataset,
                                         device=self.device,
                                        )
                
            elif "mask2former" in model_name.lower():
                arch_model = "_".join(model_name.split("_")[-2:])
                self.model = Mask2Former_Swin(arch_model=arch_model,  
                                              dataset=self.dataset,
                                              device=self.device,
                                             )
                
            elif "psp_resnet50_dilated" == model_name.lower(): # ONLY ADE20K PYTORCH
                self.model = PSP_ResNet50_Dilated(model_path=self.model_path,  
                                                  dataset=self.dataset,
                                                  device=self.device,
                                                 )
                
            elif "deeplabv3_xception65" == model_name.lower(): # ONLY ADE20K KERAS
                from .deeplabv3_xception65 import DeepLabV3_Xception65
                self.model = DeepLabV3_Xception65()
            
            else:
                self.model = None
            
        else:
            self.model = None
            
        self.model_name = self.get_model_name()
    
    def get_objects_ratio(self, masks):
        total_pixels = masks.shape[0]*masks.shape[1]
        flattened = [str(element) for row in masks for element in row]
        count_dict = Counter(flattened)
        count_dict = { k: v/total_pixels*100  for k, v  in count_dict.items() }
        return count_dict
        
    def get_objects_summary(self, idx_list, mask, total_pixel, classes_df):
        c_df = classes_df[classes_df["class_id"].isin(idx_list)].copy()
        c_df["ratio"] = c_df["class_id"].apply(lambda x: np.sum(mask==x)/total_pixel*100 )
        c_df.sort_values(by="ratio", inplace=True, ascending=False)
        
        #return c_df.to_dict(orient='records')
        #pd.DataFrame(data=objects)
        return c_df
    
    def process_masks(self, image, mask_matrix, classes_df, alpha=0.6):
        # Convert the segmentation map to RGB using the palette
        color_seg = np.zeros((mask_matrix.shape[0], mask_matrix.shape[1], 3), dtype=np.uint8)
    
        for label, color in enumerate(classes_df["RGB_color"].values):
            color_seg[mask_matrix == label+1, :] = color

        # Resize the segmentation map to match the size of the original image
        seg_img = Image.fromarray(color_seg).resize(image.size, Image.NEAREST)

        # Convert the original image to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Blend the original image with the resized segmentation map
        image_overlay = Image.blend(image, seg_img, alpha=alpha)
        
        return seg_img, image_overlay
        
    def zeroshot_segmentation(self, image, classes_df, alpha=0.6):
        mask_matrix = self.model.zeroshot_segmentation(image)
        
        seg_img, image_overlay = self.process_masks(
                                   image, 
                                   mask_matrix, 
                                   classes_df, 
                                   alpha=alpha
                                 )
        
        # mask & object dict
        unique_labels = np.unique(mask_matrix).tolist()
        w, h = image.size
        objects_summary = self.get_objects_summary(
                            unique_labels, 
                            mask_matrix, 
                            w*h, 
                            classes_df
                          )
        
        return objects_summary, mask_matrix, seg_img, image_overlay
        
