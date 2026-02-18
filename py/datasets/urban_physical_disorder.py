import ast
import json
import pandas as pd
import numpy as np
from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt

class UrbanPhysicalDisorder(): # disorder objects

    def __init__(self, 
                 data_path=None, 
                 columns_list=["image_id", "seg_image_path", "seg_overlay_image_path", "mask_path", "ratio_path"],
                ):
                
        self.data_path = data_path
        self.ade20k_path = f"{self.data_path}/ADE20k/"
        self.upd4k_path = f"{self.data_path}/UPD4k/"
        self.cityscapes_path = f"{self.data_path}/CityScapes/"
        
        self.columns_list = columns_list

    def process(self, data_df,
                      filter_features=False, 
                      keep_disorder=False,
                      threshold=0.05, 
                      aggregate_classes=False,
                      binarize_values=False):
        
        input_df = data_df.copy()
        
        if "upd4k" in self.dataset_used.lower() and keep_disorder:
            columns_to_keep = self.columns_list + self.upd4k_labels
            cur_upd4k = input_df[columns_to_keep].copy()
            input_df.drop(columns=self.upd4k_labels, errors='ignore', inplace=True)
        
        if filter_features:
            input_df = self.filter_features(input_df, threshold=threshold)

        if aggregate_classes:
            input_df = self.aggregate_classes_by_groups(input_df)
            
        if "upd4k" in self.dataset_used.lower() and keep_disorder:
            input_df = pd.merge(input_df, cur_upd4k, how="inner", on=self.columns_list)

        if binarize_values:
            input_df = self.binarize_vectors(input_df)
        
        return input_df
    
    def get_groups(self):
        if "ade20k" in self.dataset_used.lower():
            return self.ade20k_groups
        elif "cityscapes" in self.dataset_used.lower():
            return self.cityscapes_groups
        else:
            return None

    def get_urban_street_categories(self, by_groups=False):
        if by_groups:
            return self.street_categories_group()
        else:
            return self.objects_df
    
    def street_categories_group(self):
        street_df = self.objects_df[:self.dataset_len].copy()
        if "upd4k" in self.dataset_used.lower():
            upd_df = self.objects_df[self.dataset_len:].copy()
            upd_first_rows = upd_df.groupby('group_name', as_index=False).first().copy()
            upd_class_ids = upd_df.groupby('group_name')['class_id'].agg(list).reset_index()
            upd_df = pd.merge(upd_first_rows.drop(columns='class_id'), upd_class_ids, on='group_name')
            upd_df.rename(columns={"class_id": "classes"}, inplace=True)
            upd_df["group_class_id"] = upd_df["classes"].apply(lambda x: x[0])
            upd_df["num_classes"] = upd_df["classes"].apply(lambda x: len(x))

        first_rows = street_df.groupby('group_name', as_index=False).first().copy()
        class_ids = street_df.groupby('group_name')['class_id'].agg(list).reset_index()

        # Merge both
        group_df = pd.merge(first_rows.drop(columns='class_id'), class_ids, on='group_name')
        group_df.rename(columns={"class_id": "classes"}, inplace=True)
        group_df["num_classes"] = group_df["classes"].apply(lambda x: len(x))
        group_df["group_class_id"] = range(1,len(group_df)+1)

        group_df['RGB_color'] = group_df['RGB_color'].apply(str)

        group_df.loc[(group_df["group_name"]=="construction"), "RGB_color"] = '(180, 120, 120)'
        group_df.loc[(group_df["group_name"]=="construction"), "hex_color"] = "#B47878"

        group_df.loc[(group_df["group_name"]=="floor"), "RGB_color"] = '(140, 140, 140)'
        group_df.loc[(group_df["group_name"]=="floor"), "hex_color"] = "#8C8C8C"

        group_df.loc[(group_df["group_name"]=="city_elements"), "RGB_color"] = '(0, 71, 255)'
        group_df.loc[(group_df["group_name"]=="city_elements"), "hex_color"] = "#0047FF"

        group_df.loc[(group_df["group_name"]=="terrain_vehicle"), "RGB_color"] = '(255, 0, 20)'
        group_df.loc[(group_df["group_name"]=="terrain_vehicle"), "hex_color"] = "#FF0014"

        group_df['RGB_color'] = group_df['RGB_color'].apply(ast.literal_eval)
        
        if "upd4k" in self.dataset_used.lower():
            return pd.concat([group_df, upd_df], ignore_index=True)
        else:
            return group_df

    def generate_dataset(self, dataset=None):
        if "ade20k" in dataset.lower():
            self.read_ade20k_classes()
            self.dataset_len = 150
            if "upd4k" in dataset.lower():
                self.read_upd4k_classes()
                upd4k_df_ = self.upd4k_df.copy()
                upd4k_df_["class_id"] = (upd4k_df_.index + 151).tolist()
                objects_df = pd.concat([self.ade20k_df, upd4k_df_], ignore_index=True)
                self.dataset_used = "ade20k_upd4k"
                self.ade20k_labels = self.ade20k_df["main_class"].tolist()
                self.ade20k_groups = self.ade20k_df.set_index('main_class')['group_name'].to_dict()
                self.upd4k_labels = self.upd4k_df["main_class"].tolist()
            
            else:
                objects_df = self.ade20k_df.copy()
                self.dataset_used = "ade20k"
                self.ade20k_labels = self.ade20k_df["main_class"].tolist()
                self.ade20k_groups = self.ade20k_df.set_index('main_class')['group_name'].to_dict()
        
        elif "cityscapes" in dataset.lower():
            seg_path = f"{self.data_path}/cityscapes/"
            self.read_cityscapes_classes(seg_path)
            self.dataset_len = 18
            if "upd4k" in dataset.lower():
                upd4k_df_ = self.upd4k_df.copy()
                upd4k_df_["class_id"] = (upd4k_df_.index + 19).tolist()
                objects_df = pd.concat([self.cityscapes_df, upd4k_df_], ignore_index=True)
                self.dataset_used = "cityscapes_upd4k"
                self.cityscapes_labels = self.cityscapes_df["main_class"].tolist()
                self.cityscapes_groups = self.cityscapes_df.set_index('main_class')['group_name'].to_dict()
                self.upd4k_labels = self.upd4k_df["main_class"].tolist()
        
            else:
                objects_df = self.cityscapes_df.copy()
                self.dataset_used = "cityscapes"
                self.cityscapes_labels = self.cityscapes_df["main_class"].tolist()
                self.cityscapes_groups = self.cityscapes_df.set_index('main_class')['group_name'].to_dict()
            
        else:
            self.read_ade20k_classes(seg_path)
            self.dataset_len = 150
            objects_df = self.ade20k_df.copy()
            self.dataset_used = "ade20k"
            self.ade20k_labels = self.ade20k_df["main_class"].tolist()
            self.ade20k_groups = self.ade20k_df.set_index('main_class')['group_name'].to_dict()
        
        #objects_df['RGB_color'] = objects_df['RGB_color'].apply(ast.literal_eval)
        
        self.objects_df = objects_df.copy()
        self.color_dict = self.objects_df.set_index('main_class')['hex_color'].to_dict()
        self.color_dict.update(self.color_group_dict)
        
    
    def read_cityscapes_classes(self):
        
        cityscapes_df = pd.read_csv(f"{self.cityscapes_path}cityscapes_categories.csv", sep=";", low_memory=False)
        cityscapes_df = cityscapes_df[["class_id", "main_class", "class_name", "RGB_color", "hex_color", "isthing", "group_name"]].copy()
        cityscapes_df["class_id"] = cityscapes_df["class_id"].apply(lambda x: int(x) )

        cityscapes_df['RGB_color'] = cityscapes_df['RGB_color'].apply(ast.literal_eval)
        self.cityscapes_df = cityscapes_df.copy()
        
    def read_ade20k_classes(self):
        
        ade20k_groups_df = pd.read_csv(f"{self.ade20k_path}ade20k_categories_groups.csv", sep=";", low_memory=False)
        ade20k_groups_df['RGB_color'] = ade20k_groups_df['RGB_color'].apply(ast.literal_eval)
        self.ade20k_groups_df = ade20k_groups_df.copy()
        
        self.color_group_dict = self.ade20k_groups_df.set_index('group_name')['hex_color'].to_dict()
        
        ade20k_df = pd.read_csv(f"{self.ade20k_path}ade20k_categories.csv", sep=";", low_memory=False)
        ade20k_df = ade20k_df[["class_id", "main_class", "class_name", "RGB_color", "hex_color", "isthing", "group_name"]].copy()
        ade20k_df["class_id"] = ade20k_df["class_id"].apply(lambda x: int(x) )

        ade20k_df['RGB_color'] = ade20k_df['RGB_color'].apply(ast.literal_eval)
        self.ade20k_df = ade20k_df.copy()
    
    def read_ade20k_isthings(self):
        with open(f"{self.ade20k_path}ade20k_panoptic.json", "r") as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict(data, orient="index").reset_index()
        df.rename(columns={"index": "class_id"}, inplace=True)
        df.drop(columns=["name"], inplace=True)
        df["class_id"] = df["class_id"].apply(lambda x: int(x) +1 )
        self.ade20k_panoptic_df = df.copy()
    
    def create_ade20k_classes(self):
        
        self.read_ade20k_isthings(self.ade20k_path)
    
        ade20k_df = pd.read_excel(f"{self.ade20k_path}color_coding_classes.ods", engine="odf")
        ade20k_df["main_class"] = ade20k_df["class_name"].apply(lambda x: x.split(';')[0])
        ade20k_df.loc[(ade20k_df["class_id"]==59), "main_class"] = "screen_door"
        ade20k_df.loc[(ade20k_df["class_id"]==131), "main_class"] = "screen_projection"
        ade20k_df.loc[(ade20k_df["class_id"]==49), "RGB_color"] = "(140, 140, 200)"
        ade20k_df.loc[(ade20k_df["class_id"]==49), "hex_color"] = "#8C8CC8"
        ade20k_df = ade20k_df[["class_id", "main_class", "class_name", "RGB_color", "hex_color"]].copy()

        ade20k_df = pd.merge(ade20k_df, self.ade20k_panoptic_df, on="class_id", how="left")
        ade20k_df.to_csv(f"{self.ade20k_path}ade20k_color_classes.csv", sep=";", index=False)
        
        self.ade20k_df = ade20k_df.copy()
    
    def read_upd4k_classes(self):
        
        labelmap_path = f"{self.upd4k_path}/labelmap.txt"
        
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
        upd4k_df = pd.DataFrame(data, columns=["main_class", "RGB_color", "hex_color", "isthing"])
        upd4k_df = upd4k_df.iloc[1:, :].reset_index(drop=True).copy()
        upd4k_df["class_id"] = (upd4k_df.index).tolist()
        upd4k_df["class_name"] = upd4k_df["main_class"]
        
        upd4k_df['RGB_color'] = upd4k_df['RGB_color'].apply(ast.literal_eval)
        self.upd4k_df = upd4k_df.copy()
        self.upd4k_df["group_name"] = self.upd4k_df["main_class"]
    
    def filter_features(self, input_df, threshold=0.05):
        _df = input_df.copy()
        _df.drop(columns=self.columns_list, errors='ignore', inplace=True)
        
        _df[_df > 0] = 1
        _df = (_df.sum()/len(_df) ).reset_index()
        _df.sort_values(by=0, ascending=False, inplace=True)
        
        if 0>=threshold:
            df_grouped = _df.copy()
            features_to_keep = df_grouped["index"].tolist()
        
        elif 1>threshold:
            df_grouped = _df[_df[0]>= threshold].copy()
            features_to_keep = df_grouped["index"].tolist()
        
        else:
            df_grouped = _df[:threshold].copy()
            features_to_keep = df_grouped["index"].tolist()
        
        r_df= input_df[self.columns_list+features_to_keep].copy()
        return r_df
    
    def aggregate_classes_by_groups(self, input_df, keep_disorder=False):
        df_to_group = input_df.copy()
        df_to_group.drop(columns=self.columns_list, errors='ignore', inplace=True)
        
        class_to_group =  self.get_groups() 
        
        mapped_column_names = [class_to_group.get(col, col) for col in df_to_group.columns]
        
        df_to_group.columns = mapped_column_names
        
        # deprecated pandas <=2.1.0
        # df_grouped = df_to_group.groupby(df_to_group.columns, axis=1).sum()
        df_grouped = df_to_group.T.groupby(df_to_group.columns).sum().T
        
        grouped_df= pd.concat([input_df[self.columns_list].copy(), df_grouped], axis=1)
        return grouped_df
    
    def binarize_vectors(self, input_df):
        features_df = input_df.copy()
        features_df.drop(columns=self.columns_list, errors='ignore', inplace=True)
        features_df = (features_df > 0).astype(int)
        
        binarize_df = pd.concat([input_df[self.columns_list].copy(), features_df], axis=1)
        return binarize_df
        
    def features_presence(self, df):
        features_df = df.copy()
        features_df[features_df > 0] = 1
        features_df = (features_df.sum()/len(features_df) ).reset_index()
        features_df.sort_values(by=0, ascending=False, inplace=True)

        return features_df
    
    def convert_matrix_to_mask(self, input_matrix, colormap):
        height, width = input_matrix.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        for value, rgb_color in colormap.items():
            rgb_image[input_matrix == value] = rgb_color

        rgb_image = Image.fromarray(rgb_image)
        return rgb_image
    
    def convert_mask_to_matrix(self, manual_mask, manual_masks_tuple, objects_df):
        # map colors
        color_map = objects_df[objects_df["RGB_color"].isin(manual_masks_tuple)].set_index("RGB_color")["class_id"].to_dict()
    
        # Convert image to numpy array
        image_np = np.array(manual_mask)
        
        class_matrix = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.int64)

        # Assign class indices based on color mapping
        for color, class_id in color_map.items():
            mask = np.all(image_np == color, axis=-1)  # Find pixels matching the color
            class_matrix[mask] = class_id 
        
        return class_matrix
        
    def merge_segmentations(self, ade20k_mask, manual_mask):
        ade20k_mask_np = np.array(ade20k_mask)
        manual_mask_np = np.array(manual_mask)
        
        # Create a mask where mask1 is not black (i.e., contains segmented objects)
        non_black_pixels = (manual_mask_np[:, :, :3] != [0, 0, 0]).any(axis=2)
        
        # Overlay mask1 onto mask2
        merged_np = ade20k_mask_np.copy()
        merged_np[non_black_pixels] = manual_mask_np[non_black_pixels]

        # Convert back to PIL Image
        return merged_np

    def calculate_pixel_ratios(self, merged_masks, objects_df):
        unique_values, counts = np.unique(merged_masks, return_counts=True)
        total_pixels = merged_masks.size
        class_pixel_ratios = {int(val): (int(count) / total_pixels) * 100 for val, count in zip(unique_values, counts)}

        sub_df = objects_df[objects_df["class_id"].isin(unique_values)].reset_index(drop=True).copy()

        for val, count in zip(unique_values.tolist(), counts.tolist()):
            sub_df.loc[ ( sub_df["class_id"]==val ) , "ratio"] = count / total_pixels * 100
        
        return sub_df

    def calculate_pixel_ratios_from_colors(self, merged_np, objects_df):

        # calculate new pixel ratios
        total_pixels = merged_np.shape[0] * merged_np.shape[1]

        # Count occurrences of each class using the DataFrame
        class_pixel_counts = {}
        for _, row in objects_df.iterrows():
            class_pixel_counts[row['main_class']] = np.all(merged_np[..., :3] == row['RGB_color'], axis=-1).sum()
        
        # Compute pixel ratio
        class_pixel_ratios = {cls: count / total_pixels*100 for cls, count in class_pixel_counts.items()}
        
        df_ratios = pd.DataFrame.from_dict(class_pixel_ratios, orient="index").T
  
        return df_ratios
        
    def calculate_unique_colors(self, manual_mask, to_tuple=False):
        # Convert image to numpy array
        img_array = np.array(manual_mask)

        # Reshape and find unique colors
        unique_colors = np.unique(img_array.reshape(-1, 3), axis=0).tolist()
        
        if to_tuple:
            unique_colors = [tuple(color) for color in unique_colors]
        
        return unique_colors
    
    def parse_ratios(self, current_id, ratio_df):
        df_pivot = ratio_df.pivot_table(index=None, columns="main_class", values="ratio")
        df_pivot.columns.name = None
        df_pivot = df_pivot.reset_index(drop=True)
        df_pivot["image_id"] = current_id
        return df_pivot
    
    # PLOT
    def print_object_present(self, features_df, fig_size=(40,20), top_k=None):
        try:
            fig, ax = plt.subplots(figsize=fig_size, nrows=1, ncols=1, sharex=False, sharey=False)
            
            if top_k is not None:
                features_df = features_df.head(top_k).copy()
            
            sns_fig = sns.barplot(
                        data=features_df,
                        x=features_df.columns[1],
                        y=features_df.columns[0],
                        ax=ax,
                        palette=self.color_dict, 
                        # order=estado_df.sort_values(estado_df.columns[1], ascending=False)[estado_df.columns[0]]
                       )

            sns_fig.set_title(f"Elements presence", fontsize=40)
            sns_fig.set_ylabel(f"Number of comparisons", fontsize=0)
            sns_fig.set_xlabel('% of images containing this object', fontsize=40)

            # rotate the axis ticklabels
            _ = sns_fig.tick_params(axis='x', rotation=0, labelsize=45)
            
            # rotate the axis ticklabels
            _ = sns_fig.tick_params(axis='y', labelsize=40)

            # add annotation
            #_ = sns_fig.bar_label(sns_fig.containers[0], fmt='%0.0f', fontsize=15,rotation=0)

            # add a space on y for the annotations
            #sns_fig.margins(x=0.1)
            ax.grid(True)

            plt.show()
        except Exception as e:
            print(e)
    
    def print_object_boxplot(self, features_df, fig_size=(40,20), top_k=None, pixel_presence=True):
        try:
            fig, ax = plt.subplots(figsize=fig_size, nrows=1, ncols=1, sharex=False, sharey=False, constrained_layout=True)
            
            sorted_columns = features_df.median().sort_values(ascending=False).index.tolist()
            
            if top_k is not None:
                sorted_columns = sorted_columns[:top_k].copy()
            
            if pixel_presence:
                features_df[sorted_columns] = features_df[sorted_columns]/100
        
            sns_fig = sns.boxplot(
                        data=features_df[sorted_columns],
                        orient="h",
                        # dodge=True,
                        flierprops=dict(marker='o', markersize=30),
                        # showfliers=True, 
                        medianprops=dict(color="green", linewidth=5),
                
                        notch=True, showcaps=False,
                        color="steelblue",
                        native_scale=True,
                        palette=self.color_dict,
                        # order=estado_df.sort_values(estado_df.columns[1], ascending=False)[estado_df.columns[0]]
                       )
            
            sns_fig.set_title(f"Object presence", fontsize=50)
            sns_fig.set_ylabel(f"", fontsize=0)
            sns_fig.set_xlabel('% of pixels within images', fontsize=50)
            
            # rotate the axis ticklabels
            _ = sns_fig.tick_params(axis='x', rotation=0, labelsize=45)
            
            # rotate the axis ticklabels
            _ = sns_fig.tick_params(axis='y', labelsize=40)
            
            #sns_fig.set_xticks(list(range(10)))
            #sns_fig.set_xticklabels([str(i) for i in list(range(10))])
            
            # add a space on y for the annotations
            sns_fig.margins(x=0.1)
            ax.grid(True)
            
            plt.show()
        except Exception as e:
            print(e)

