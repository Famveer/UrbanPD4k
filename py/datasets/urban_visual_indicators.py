class UrbanVisualIndicators:

    def __init__(self, source_format="mask"):
        self.source_format = source_format
        self.vegetation_list = ["grass", "field", "bush", "plant", "leaf", "flower", "vine", "moss", "crop", "palm", "terrain" ]
        self.earth_list = ['dirt', 'sand', 'rock', 'mud', 'gravel', 'snow', 'ice', 'leaf-litter', "path"]
        self.pavement_list = ['sidewalk', 'pavement','street', 'parking_lot', 'pathway', 'driveway', 'plaza', 'tennis_court']
        self.guardrail_list = ['guardrail', 'handrail', 'rail', 'railing', 'balustrade', 'barrier']


        #self.tree_mask = self.mask[self.mask["names"]=="tree"]["masks"].values[0]
        #shape_mask = np.prod(self.tree_mask.shape)
        
    def calculate_visual_indicators(self, mask):
        vi_dict = {}
        
        vi_dict["greenness"] = self.Greenness(mask)
        vi_dict["openness"] = self.Openness(mask)
        vi_dict["enclosure"] = self.Enclosure(mask)
        vi_dict["safety"] = self.Safety(mask)
        vi_dict["walkability"] = self.Walkability(mask)
        vi_dict["imageability"] = self.Imageability(mask)
        vi_dict["complexity"] = self.Complexity(mask)
        
        return vi_dict

    def eval_ratio(self, value):
        return 1 if value <= 0.099 else value
        
    def View_of_Index(self, mask, name):
        
        if self.source_format == "csv":
            return self.View_of_Index_from_csv(mask, name)
        if self.source_format == "mask":
            return self.View_of_Index_from_mask(mask, name)
        else:
            return self.View_of_Index_from_csv(mask, name)

    def View_of_Index_from_mask(self, mask, name):
        
        info_ = mask[mask["main_class"]==name]["ratio"].values
        
        if len(info_) > 0:
            return info_[0]/100
        else:
            return 0
    
    def View_of_Index_from_csv(self, mask, name):
        
        try:
            info_ = mask[name].values
        except Exception as e:
            info_=[]
        
        if len(info_) > 0:
            return info_[0]/100
        else:
            return 0

    def Greenness(self, mask):
        """
        Calculate or evaluate the greenness of an urban area.
        This could involve analyzing vegetation cover, tree density, etc.
        """
        # Implement the logic for greenness evaluation
        VoI_vegetation = 0
        for obj_name in self.vegetation_list:
            VoI_vegetation += self.View_of_Index(mask, obj_name)
        
        return self.View_of_Index(mask, "tree") + VoI_vegetation


    def Openness(self, mask): 
        """
        Calculate or evaluate the openness of an urban area.
        This could involve assessing open spaces, parks, plazas, etc.
        """
        # Implement the logic for openness evaluation
        return self.View_of_Index(mask, "sky")


    def Enclosure(self, mask):
        """
        Calculate or evaluate the enclosure of an urban area.
        This could involve analyzing the sense of being surrounded by buildings, walls, etc.
        """
        # Implement the logic for enclosure evaluation
        
        VoI_road_sidewalk_fence = self.View_of_Index(mask, "road") + self.View_of_Index(mask, "sidewalk") + self.View_of_Index(mask, "fence")
        
        denominator = self.eval_ratio(VoI_road_sidewalk_fence)
        
        result =  (self.View_of_Index(mask, "tree") + self.View_of_Index(mask, "building") + self.View_of_Index(mask, "wall")) / (denominator)
        
        return 1.0 if result >= 1 else result
        

    def Safety(self, mask):
        """
        Calculate or evaluate the safety of an urban area.
        This could involve crime statistics, lighting, surveillance, etc.
        """
        # Implement the logic for safety evaluation
        VoI_guardrail = 0
        for obj_name in self.guardrail_list:
            VoI_guardrail += self.View_of_Index(mask, obj_name)
        
        return self.View_of_Index(mask, "person") + self.View_of_Index(mask, "rider") + self.View_of_Index(mask, "signboard") + self.View_of_Index(mask, "light") + self.View_of_Index(mask, "streetlight") + self.View_of_Index(mask, "fence") + self.View_of_Index(mask, "streetlight") + VoI_guardrail


    def Walkability(self, mask):
        """
        Calculate or evaluate the walkability of an urban area.
        This could involve the quality of sidewalks, pedestrian crossings, distance to amenities, etc.
        """
        # Implement the logic for walkability evaluation
        VoI_pavement = 0
        for obj_name in self.pavement_list:
            VoI_pavement += self.View_of_Index(mask, obj_name)
            
        VoI_road = self.View_of_Index(mask, "road")
        denominator = self.eval_ratio(VoI_road)
        
        result = (VoI_pavement + self.View_of_Index(mask, "fence")) / (denominator)
        
        return 1.0 if result >= 1 else result


    def Imageability(self, mask):
        """
        Calculate or evaluate the imageability of an urban area.
        This could involve landmarks, architectural style, distinctiveness, etc.
        """
        # Implement the logic for imageability evaluation
        return self.View_of_Index(mask, "skyscraper") + self.View_of_Index(mask, "building") + self.View_of_Index(mask, "signboard") 


    def Complexity(self, mask):
        """
        Calculate or evaluate the complexity of an urban area.
        This could involve the variety of building designs, street patterns, visual richness, etc.
        """
        # Implement the logic for complexity evaluation
        
        VoI_earth = 0
        for obj_name in self.earth_list:
            VoI_earth += self.View_of_Index(mask, obj_name)
        
        VoI_road_build = self.View_of_Index(mask, "road") + self.View_of_Index(mask, "building") + VoI_earth
        denominator = self.eval_ratio(VoI_road_build)
        
        result = ( self.View_of_Index(mask, "person") + self.View_of_Index(mask, "rider") + self.View_of_Index(mask, "signboard") + self.View_of_Index(mask, "light") + self.View_of_Index(mask, "streetlight")  + self.View_of_Index(mask, "tree") ) / (denominator)
        
        return 1.0 if result >= 1 else result
        
