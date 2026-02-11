import torch
import torch.nn as nn

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

class OneFormer_Swin(nn.Module):
    def __init__(self, device, arch_model="large", dataset="ade20k"):
        super(OneFormer_Swin, self).__init__()
        self.model_name = f"OneFormer_{arch_model.capitalize()}"
        self.dataset = dataset
        self.device = device

        if arch_model.lower()=="swin_large":
            if dataset == "ade20k":
                self.model_arch = "shi-labs/oneformer_ade20k_swin_large"
            elif dataset == "cityscapes":
                self.model_arch = "shi-labs/oneformer_cityscapes_swin_large"
            else:
                self.model_arch = "shi-labs/oneformer_ade20k_swin_large"
            
        elif arch_model.lower()=="swin_tiny":
            self.model_arch = "shi-labs/oneformer_ade20k_swin_tiny"
            
        elif arch_model.lower()=="dinat_large":
            if dataset == "ade20k":
                self.model_arch = "shi-labs/oneformer_ade20k_dinat_large"
            elif dataset == "cityscapes":
                self.model_arch = "shi-labs/oneformer_cityscapes_dinat_large"
            else:
                self.model_arch = "shi-labs/oneformer_ade20k_dinat_large"
            
        else:
            if dataset == "ade20k":
                self.model_arch = "shi-labs/oneformer_ade20k_swin_large"
            elif dataset == "cityscapes":
                self.model_arch = "shi-labs/oneformer_cityscapes_swin_large"
            else:
                self.model_arch = "shi-labs/oneformer_ade20k_swin_large"
            
        self.processor = OneFormerProcessor.from_pretrained(
                                            self.model_arch, 
                                            use_fast=False,
                                            )
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
                        self.model_arch, 
                        )

    def get_processor(self):
        return self.processor
        
    def get_model_arch(self):
        return self.model_arch
        
    def get_model_name(self):
        return self.model_name
    
    def freeze(self):
        for param in model.parameters():
            param.requires_grad = False
    
    def forward(self):
        pass
        
    def zeroshot_segmentation(self, image):
        semantic_inputs = self.processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(self.device)
        
        #if torch.cuda.is_available():
        #    semantic_inputs = {k: v.cuda() for k, v in semantic_inputs.items()}
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(**semantic_inputs)
            
        # Get the class with the highest score at each pixel (predicted segmentation map)
        color_matrix = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()
        
        mask_matrix = color_matrix+1
        return mask_matrix
