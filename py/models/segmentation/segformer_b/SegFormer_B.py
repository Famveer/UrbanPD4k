import torch
import torch.nn as nn
import cv2

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class SegFormer_B(nn.Module):
    def __init__(self, device, arch_model="b5", dataset="ade20k"):
        super(SegFormer_B, self).__init__()
        self.model_name = f"SegFormer_{arch_model.capitalize()}"
        self.dataset = dataset
        self.device = device
        
        if arch_model.lower()=="b0":
            if dataset == "ade20k":
                self.model_arch = "nvidia/segformer-b0-finetuned-ade-512-512"
            elif dataset == "cityscapes":
                self.model_arch = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
            else:
                self.model_arch = "nvidia/segformer-b0-finetuned-ade-512-512"
            
        elif arch_model.lower()=="b1":
            if dataset == "ade20k":
                self.model_arch = "nvidia/segformer-b1-finetuned-ade-512-512"
            elif dataset == "cityscapes":
                self.model_arch = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
            else:
                self.model_arch = "nvidia/segformer-b1-finetuned-ade-512-512"
            
        elif arch_model.lower()=="b2":
            if dataset == "ade20k":
                self.model_arch = "nvidia/segformer-b2-finetuned-ade-512-512"
            elif dataset == "cityscapes":
                self.model_arch = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
            else:
                self.model_arch = "nvidia/segformer-b2-finetuned-ade-512-512"
            
        elif arch_model.lower()=="b3":
            if dataset == "ade20k":
                self.model_arch = "nvidia/segformer-b3-finetuned-ade-512-512"
            elif dataset == "cityscapes":
                self.model_arch = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
            else:
                self.model_arch = "nvidia/segformer-b3-finetuned-ade-512-512"
            
        elif arch_model.lower()=="b4":
            if dataset == "ade20k":
                self.model_arch = "nvidia/segformer-b4-finetuned-ade-512-512"
            elif dataset == "cityscapes":
                self.model_arch = "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"
            else:
                self.model_arch = "nvidia/segformer-b4-finetuned-ade-512-512"
            
        elif arch_model.lower()=="b5":
            if dataset == "ade20k":
                self.model_arch = "nvidia/segformer-b5-finetuned-ade-640-640"
            elif dataset == "cityscapes":
                self.model_arch = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
            else:
                self.model_arch = "nvidia/segformer-b5-finetuned-ade-640-640"
            
        else:
            if dataset == "ade20k":
                self.model_arch = "nvidia/segformer-b5-finetuned-ade-640-640"
            elif dataset == "cityscapes":
                self.model_arch = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
            else:
                self.model_arch = "nvidia/segformer-b5-finetuned-ade-640-640"
        
        self.processor = SegformerImageProcessor.from_pretrained(self.model_arch)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
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
        
        with torch.no_grad():
            outputs = self.model(**semantic_inputs)
            
        # Get the segmentation map (logits)
        logits = outputs.logits  # Shape: [batch_size, num_classes, height, width]

        # Get the class with the highest score at each pixel (predicted segmentation map)
        color_matrix = torch.argmax(logits, dim=1).squeeze().cpu().numpy()  # Shape: [height, width]
        
        mask_matrix = cv2.resize(color_matrix, image.size, interpolation=cv2.INTER_NEAREST)+1
        
        return mask_matrix
