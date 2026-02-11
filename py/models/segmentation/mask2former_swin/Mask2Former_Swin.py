import torch
import torch.nn as nn

from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

class Mask2Former_Swin(nn.Module):
    def __init__(self, device, arch_model="large", dataset="ade20k"):
        super(Mask2Former_Swin, self).__init__()
        self.model_name = f"Mask2Former_{arch_model.capitalize()}"
        self.dataset = dataset
        self.device = device
        
        if arch_model.lower()=="swin_base":
            self.model_arch = "facebook/mask2former-swin-base-ade-semantic"
        
        elif arch_model.lower()=="swin_large":
            if dataset == "ade20k":
                self.model_arch = "facebook/mask2former-swin-large-ade-semantic"
            elif dataset == "cityscapes":
                self.model_arch = "facebook/mask2former-swin-large-cityscapes-semantic"
            else:
                self.model_arch = "facebook/mask2former-swin-large-ade-semantic"
            
        elif arch_model.lower()=="swin_small":
            if dataset == "ade20k":
                self.model_arch = "facebook/mask2former-swin-small-ade-semantic"
            elif dataset == "cityscapes":
                self.model_arch = "facebook/mask2former-swin-small-cityscapes-semantic"
            else:
                self.model_arch = "facebook/mask2former-swin-small-ade-semantic"
            
        elif arch_model.lower()=="swin_tiny":
            if dataset == "ade20k":
                self.model_arch = "facebook/mask2former-swin-tiny-ade-semantic"
            elif dataset == "cityscapes":
                self.model_arch = "facebook/mask2former-swin-tiny-cityscapes-semantic"
            else:
                self.model_arch = "facebook/mask2former-swin-tiny-ade-semantic"
            
        else:
            if dataset == "ade20k":
                self.model_arch = "facebook/mask2former-swin-large-ade-semantic"
            elif dataset == "cityscapes":
                self.model_arch = "facebook/mask2former-swin-large-cityscapes-semantic"
            else:
                self.model_arch = "facebook/mask2former-swin-large-ade-semantic"
            
        self.processor = Mask2FormerImageProcessor.from_pretrained(self.model_arch)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
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
        semantic_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        #if torch.cuda.is_available():
        #    semantic_inputs = {k: v.cuda() for k, v in semantic_inputs.items()}
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(**semantic_inputs)
            
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits
            
        # Get the class with the highest score at each pixel (predicted segmentation map)
        color_matrix = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()
        
        mask_matrix = color_matrix+1
        return mask_matrix
