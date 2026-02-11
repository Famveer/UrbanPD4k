import pixellib
from pixellib.semantic import semantic_segmentation, labelAde20k_to_color_image
import cv2
from PIL import Image
import numpy as np
from scipy.io import loadmat
import csv

class DeepLabV3_Xception65():

    def __init__(self, MODEL_PATH):
    
        self.model_name = "DeepLabV3_Xception65"
        self.colors_path = f"{MODEL_PATH}categories/color150.mat"
        self.names_path = f"{MODEL_PATH}categories/object150.csv"
        self.model_path = f"{MODEL_PATH}deeplabv3_xception65/deeplabv3_xception65_ade20k.h5"

        self.names, self.colors = self.get_ade20k_labels()
    
        self.initialize_model()
    
    def initialize_model(self):
        segment_image = semantic_segmentation()
        segment_image.load_ade20k_model(f"{self.model_path}")
        self.model = segment_image.model2
        
    def get_ade20k_labels(self):
        colors = loadmat(self.colors_path)['colors']
        colors = np.concatenate([np.zeros(shape=(1,3)), colors])
        names = {0: "no class"}
        with open(self.names_path) as f:
          reader = csv.reader(f)
          next(reader)
          for row in reader:
            names[int(row[0])] = row[5].split(";")[0]
        
        names[59] = "screen_door"
        names[131] = "screen_projection"
        
        self.name_colors_dict = dict(zip( list(names.values()), colors ))

        return names, colors
        
    def to_device(self, device="cpu"):
        self.model = self.model.to(device)

    def get_model(self):
        return self.model
    
    def get_model_name(self):
        return self.model_name
      
    def process_segmentation(self, image_path, alpha = 0.6):

        trained_image_width=512
        mean_subtraction_value=127.5
        image = np.array(Image.open(image_path))
        
        # resize to max dimension of images from training dataset
        w, h, n = image.shape

        if n > 3:
          image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        ratio = float(trained_image_width) / np.max([w, h])
        resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
        resized_image = (resized_image / mean_subtraction_value) -1


        # pad array to square image to match training images
        pad_x = int(trained_image_width - resized_image.shape[0])
        pad_y = int(trained_image_width - resized_image.shape[1])
        resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

        #run prediction
        res = self.model.predict(np.expand_dims(resized_image, 0))
        
        labels = np.argmax(res.squeeze(), -1)
        
        # remove padding and resize back to original image
        if pad_x > 0:
          labels = labels[:-pad_x]
        if pad_y > 0:
          labels = labels[:, :-pad_y]

        """Run here the new function"""
        _, objects = self.process_masks(labels)
        
        """ Access the unique class ids of the masks """
        color_matrix = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
        
        """ Convert indexed masks to boolean """
        #color_matrix = np.ma.make_mask(color_matrix)
        mask_seg_values = color_matrix
       
        #Apply segmentation color map
        labels = labelAde20k_to_color_image(labels)   
        seg_img = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
        #seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
        
        image_overlay = image.copy()
        #image_overlay = seg_img * alpha + image_overlay*(1-alpha)
        image_overlay = cv2.addWeighted(seg_img, alpha, image_overlay, 1 - alpha,0)
        
        return objects, mask_seg_values, Image.fromarray(seg_img), Image.fromarray(image_overlay)

    def process_masks(self, raw_mask):
        uniques, counts = np.unique(raw_mask, return_counts=True)
        
        class_index = []
        masks = []
        ratios = []
        class_name = []
        class_color = []

        d_dict = []

        for idx in np.argsort(counts)[::-1]:
            index_label = uniques[idx]
            label_mask = raw_mask == index_label

            class_index.append(index_label)
            masks.append(label_mask)
            ratios.append(counts[idx]/raw_mask.size *100)
            class_name.append(self.names[index_label])
            class_color.append(self.colors[index_label].tolist())

            d_dict.append({"class_id": index_label,
                            "class_name": self.names[index_label], 
                            "RGB_color": self.colors[index_label].tolist(), 
                            #"masks": label_mask, 
                            "ratio": counts[idx]/raw_mask.size *100})
        
        d_segment = {"class_id": class_index,
                     "class_name": class_name, 
                     "RGB_color": class_color, 
                     #"masks": masks, 
                     "ratio": ratios}

        return d_segment, d_dict
