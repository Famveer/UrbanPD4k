import torch
import torch.nn as nn
import numpy as np

from .lib.networks import PSPNetForSemanticSegmentation, PSPNetImageProcessor

class PSP_ResNet50_Dilated(nn.Module):
    def __init__(self, device, model_path):
        super(PSP_ResNet50_Dilated, self).__init__()
        self.model_name = "PSP_ResNet50_Dilated"
        self.model_arch = "shuangzhao/pspnet-resnet50-ade-dilated"
        self.device = device
        
        encoder_path = f"{model_path}/SceneParser_ADE20K/pspnet_resnet50/encoder_epoch_20.pth"
        decoder_path = f"{model_path}/SceneParser_ADE20K/pspnet_resnet50/decoder_epoch_20.pth"
        
        self.processor = PSPNetImageProcessor.from_pretrained(
                        self.model_arch,
                        device=self.device,
                        )
        self.model = PSPNetForSemanticSegmentation.from_pretrained(
                        self.model_arch, 
                        encoder_path=encoder_path,
                        decoder_path=decoder_path,
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
        
    def zeroshot_segmentation(self, image, alpha = 0.6):
        h, w  = image.size
        
        img_resized_list = self.processor(image)
        
        with torch.no_grad():
            scores = torch.zeros(1, 150, w, h).to(self.device)
            
            for img in img_resized_list:
                #feed_dict = {'img_data': img}
                #feed_dict = async_copy_to(feed_dict, 0)
            
                #forward pass
                pred_tmp = self.model(img, segSize=(w, h))
                scores = scores + pred_tmp / len(self.processor.imgSizes)
        
        _, pred = torch.max(scores, dim=1)
        mask_matrix = pred.squeeze(0).cpu().numpy()
        
        # generating masks
        mask_matrix = np.int64(mask_matrix) + 1
        
        return mask_matrix
