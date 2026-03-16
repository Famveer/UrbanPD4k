import numpy as np
import pandas as pd
from PIL import Image


class MaskProcessor:
    """
    Stateless utility class for pixel-level segmentation mask operations.

    All methods are static — no instance state is needed.  Import and call
    directly:

        from mask_processor import MaskProcessor

        ratios = MaskProcessor.calculate_pixel_ratios(matrix, objects_df)
        merged = MaskProcessor.merge_masks(ade_mask, manual_mask)
    """

    # ------------------------------------------------------------------
    # Matrix ↔ RGB image conversions
    # ------------------------------------------------------------------

    @staticmethod
    def convert_matrix_to_mask(input_matrix: np.ndarray, colormap: dict) -> Image.Image:
        """
        Convert a 2-D class-ID matrix to an RGB PIL image.

        Parameters
        ----------
        input_matrix : np.ndarray, shape (H, W)
            Integer array where each value is a class ID.
        colormap : dict
            {class_id: (R, G, B)} mapping.

        Returns
        -------
        PIL.Image.Image  RGB image.
        """
        height, width = input_matrix.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        for value, rgb_color in colormap.items():
            rgb_image[input_matrix == value] = rgb_color
        return Image.fromarray(rgb_image)

    @staticmethod
    def convert_mask_to_matrix(
        manual_mask: Image.Image,
        color_tuple: list,
        objects_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Convert an RGB PIL mask image back to a 2-D class-ID matrix.

        Parameters
        ----------
        manual_mask : PIL.Image.Image
        color_tuple : list of tuple
            RGB colour tuples present in the mask.
        objects_df : pd.DataFrame
            Must contain 'RGB_color' and 'class_id' columns.

        Returns
        -------
        np.ndarray  shape (H, W), dtype int64.
        """
        color_map = (
            objects_df[objects_df["RGB_color"].isin(color_tuple)]
            .set_index("RGB_color")["class_id"]
            .to_dict()
        )
        # Convert image to numpy array
        image_np     = np.array(manual_mask)
        class_matrix = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.int64)
        # Assign class indices based on color mapping
        for color, class_id in color_map.items():
            # Find pixels matching the color
            mask = np.all(image_np == color, axis=-1)
            class_matrix[mask] = class_id

        return class_matrix

    # ------------------------------------------------------------------
    # Mask merging
    # ------------------------------------------------------------------

    @staticmethod
    def merge_masks(
        current_mask: Image.Image,
        new_mask: Image.Image,
    ) -> np.ndarray:
        """
        Overlay a manual (UPD4K) mask onto an ADE20K base mask.

        Non-black pixels in *new_mask* replace the corresponding pixels
        in *current_mask*.

        Returns
        -------
        np.ndarray  Merged RGB array (H, W, 3).
        """
        current_np = np.array(current_mask)
        new_np     = np.array(new_mask)
        non_black  = (new_np[:, :, :3] != [0, 0, 0]).any(axis=2)
        merged_np  = current_np.copy()
        merged_np[non_black] = new_np[non_black]
        return merged_np

    # ------------------------------------------------------------------
    # Pixel ratio calculation
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_pixel_ratios(
        merged_masks: np.ndarray,
        objects_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute per-class pixel coverage (%) from a class-ID matrix.
        """
        unique_values, counts = np.unique(merged_masks, return_counts=True)
        total_pixels = merged_masks.size

        sub_df = objects_df[objects_df["class_id"].isin(unique_values)].reset_index(drop=True).copy()
        for val, count in zip(unique_values.tolist(), counts.tolist()):
            sub_df.loc[sub_df["class_id"] == val, "ratio"] = count / total_pixels * 100
        return sub_df

    @staticmethod
    def calculate_pixel_ratios_from_colors(
        merged_np: np.ndarray,
        objects_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute per-class pixel coverage (%) by matching RGB colours.
        """
        total_pixels = merged_np.shape[0] * merged_np.shape[1]
        class_pixel_counts = {
            row["main_class_name"]: np.all(merged_np[..., :3] == row["RGB_color"], axis=-1).sum()
            for _, row in objects_df.iterrows()
        }
        class_pixel_ratios = {cls: count / total_pixels * 100 for cls, count in class_pixel_counts.items()}
        return pd.DataFrame.from_dict(class_pixel_ratios, orient="index").T

    # ------------------------------------------------------------------
    # Colour utilities
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_unique_colors(seg_mask: Image.Image, to_tuple: bool = False) -> list:
        """
        Return the unique RGB colours present in a mask image.
        """
        img_array    = np.array(seg_mask)
        unique_colors = np.unique(img_array.reshape(-1, 3), axis=0).tolist()
        if to_tuple:
            unique_colors = [tuple(c) for c in unique_colors]
        return unique_colors

    # ------------------------------------------------------------------
    # Ratio DataFrame helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_ratios(current_id, ratio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot a long-format ratio DataFrame to wide format and attach image_id.

        Parameters
        ----------
        current_id : any
            Identifier to attach as the 'image_id' column.
        ratio_df : pd.DataFrame
            Must contain 'main_class_name' and 'ratio' columns.

        Returns
        -------
        pd.DataFrame  One row, one column per class.
        """
        df_pivot = ratio_df.pivot_table(index=None, columns="main_class_name", values="ratio")
        df_pivot.columns.name = None
        df_pivot = df_pivot.reset_index(drop=True)
        df_pivot["image_id"] = current_id
        return df_pivot
