import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import tensorflow as tf
from packaging import version
import os
from scipy.signal import convolve2d
from tqdm import tqdm
import cv2
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Segmentation:
    def __init__(self, img, spot_center: np.ndarray, out_dir: str = '', prob_thresh: float = 0.2,
                 nms_thresh: float = 0.5, spot_radius: float = 36.5, n_tiles=(8, 8, 1), 
                 enhancement=False, enhancement_params=None):
        """
        Args:
            img (numpy.ndarray):. Three channel stained image. In default, it should be hematoxylin and eosin (H&E) stained
                image.
            spot_center: Coordinates of the center of the spots.
            out_dir: Output directory.
            Prob_thresh, nms_thresh: Two thresholds used in Stardist. User should adjust these two threshold based on
                the segmentation results.
            spot_radius: Radius of the spots. In 10X Visium, it should be 36.5.
            n_tiles: Out of memory (OOM) errors can occur if the input image is too large. To avoid this problem, the
                input image is broken up into (overlapping) tiles that are processed independently and re-assembled.
                (Copied from stardist document). In default, we break the image into 8*8 tiles.
            enhancement: Whether to apply image enhancement before segmentation. (testing)
            enhancement_params: Optional, the parameter in the enhancement procedure. If not provided, use default. (testing)

        """
        self.enhancement_params = enhancement_params if enhancement_params else {'cla_wt': 0.2}
        if enhancement:
            self.img = self._apply_enhancement(img, **self.enhancement_params)
        else:
            self.img = img

        if spot_center is not None:
            assert spot_center.shape[1] == 2, "spot_center must have exact two columns."
        self.spot_center = np.array(spot_center)
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.spot_radius = spot_radius
        if out_dir and not os.path.exists(out_dir):
            os.mkdir(out_dir)
        self.out_dir = out_dir + '/' if out_dir else ''
        # Segmentation results
        self.label, self.nucleus_boundary, self.probability = None, None, None
        self.n_cell_df = None
        self.nucleus_df = None
        self.is_segmented = False  # Whether we have conducted segmentation or not.
        self.n_tiles = n_tiles
        if version.parse(tf.__version__) >= version.parse("2.9.0"):
            tf.keras.Model.predict = change_predict_defaults(tf.keras.Model.predict)
            print(f"Suppress the output of tensorflow prediction for tensorflow version {tf.version.VERSION}>=2.9.0.")

    @staticmethod
    def _apply_enhancement(img, cla_wt: float):
        """Apply CLAHE and bilateral filtering to enhance the image.

        Args:
            img: Input RGB image.
            cla_wt: Weight for combining CLAHE and bilateral filtering results. Must be between 0 and 1.

        Returns:
            np.ndarray: Enhanced image.
        """

        def _bilateral(img):
            bilateral_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
            return bilateral_filtered

        def _clahe(img):
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            result_rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            return result_rgb

        def _CLA_BIL(img, cla_wt):
            clahe_img = _clahe(img)
            bil_img = _bilateral(img)
            combined_result = cv2.addWeighted(clahe_img, cla_wt, bil_img, 1-cla_wt, 0)
            return combined_result
        
        img = _CLA_BIL(img, cla_wt)

        return img


    @staticmethod
    def stardist_2D_versatile_he(img, prob_thresh: float = 0.2, nms_thresh: float = 0.5, n_tiles=(8, 8, 1),
                                 verbose: bool = True):
        """Segmentation function provided by Stardist.

        Args:
            img: Three channel image.
            nms_thresh: Parameter of non-maximum suppression.
            prob_thresh: The probability threshold that determines the retention of a nucleus.
            n_tiles: Out of memory (OOM) errors can occur if the input image is too large. To avoid this problem, the
                input image is broken up into (overlapping) tiles that are processed independently and re-assembled.
                (Copied from stardist document). In default, we break the image into 8*8 tiles.
            verbose: Whether to print segmentation progress.

        Returns:
            There are two returns.

            np.ndarray: The segmented image. Background pixels has value 0 and nucleus pixels has
            the positive integer value as the index of the nucleus.

            list: Nucleus details. Details[0]: np.ndarray(n_nucleus*2*32). Boundaries of each nucleus. Details[1]:
            np.ndarray(n_nucleus*2). Center of each nucleus. Details[2]: np.ndarray(n_nucleus). Probability that a
            segmented nucleus is indeed a nucleus.
        """

        axis_norm = (0, 1, 2)  # normalize channels jointly
        img = normalize(img, 1, 99.8, axis=axis_norm)
        model = StarDist2D.from_pretrained('2D_versatile_he')
        label, details = model.predict_instances(img, nms_thresh=nms_thresh, prob_thresh=prob_thresh, n_tiles=n_tiles,
                                                 show_tile_progress=verbose)
        return label, details

    @staticmethod
    def n_cell_in_spot(nucleus_center: np.ndarray, spot_center: np.ndarray, spot_radius: float,
                       nucleus_df: pd.DataFrame = None) -> pd.DataFrame:
        """Find the number of cells in each spot.

        If the center of a nucleus is inside the spot, we assume that the cell is in the spot.

        Args:
            nucleus_center: np.ndarray(n_nucleus*2). Coordinates of the nucleus centers.
            spot_center: np.ndarray(n_spot*2). Coordinates of the spot centers.
            spot_radius: Radius of the spots.
            nucleus_df: Optional, dataframe of the nucleus.

        Returns:
            Pandas data frame with two columns. Column 'cell_count' represents the number of cells in a spot.
            Column 'centers' represents the coordinates of the nucleus centers.
        """
        n_spot = len(spot_center)
        n_cell_df = pd.DataFrame({'cell_count': [0] * n_spot, 'Nucleus centers': None, 'Nucleus indices':None})
        distance = np.sum((spot_center[:, :, np.newaxis] - nucleus_center.T) ** 2, axis=1)
        if nucleus_df is not None:
            nucleus_df['in_spot'] = False
        for i in range(n_spot):
            nucleus_index = np.where(distance[i] < spot_radius ** 2)[0]
            n_cell_df.iloc[i, 0] = len(nucleus_index)
            n_cell_df.at[i, 'Nucleus centers'] = nucleus_center[nucleus_index] if len(nucleus_index) else np.array([])
            n_cell_df.at[i, 'Nucleus indices'] = nucleus_index
            if nucleus_df is not None:
                nucleus_df.loc[nucleus_index, 'in_spot'] = True
        return n_cell_df

    def segment_nucleus(self, save=True):
        """Conduct the segmentation using StarDist pretrained model.

        Args:
            save: Whether to save the segmentation results.
        """
        self.label, details = self.stardist_2D_versatile_he(self.img, nms_thresh=self.nms_thresh,
                                                            prob_thresh=self.prob_thresh, n_tiles=self.n_tiles)
        nucleus_boundary, nucleus_center, self.probability = details['coord'], details['points'], details['prob']
        nucleus_boundary = np.transpose(nucleus_boundary, [0, 2, 1])
        self.nucleus_boundary = nucleus_boundary[:, :, [1, 0]]
        self.nucleus_df = pd.DataFrame({'x': nucleus_center[:, 1], 'y': nucleus_center[:, 0]})
        self.n_cell_df = self.n_cell_in_spot(self.nucleus_df[['x', 'y']].values, self.spot_center, self.spot_radius,
                                             self.nucleus_df)
        self.is_segmented = True
        if save:
            self.save_results()

    def save_results(self):
        """Save segmentation results.
        """
        assert self.is_segmented, "Please conduct segmentation first."
        np.save(f'{self.out_dir}segmentation_label.npy', self.label)
        np.save(f'{self.out_dir}segmentation_boundary.npy', self.nucleus_boundary)
        np.save(f'{self.out_dir}nucleus_df.csv', self.nucleus_df)
        np.save(f'{self.out_dir}segmentation_probability.npy', self.probability)
        self.n_cell_df.to_csv(f'{self.out_dir}n_cell_df.csv')

    def plot(self, fig_size=(10, 4.5), dpi=300, crop=None, cmap_segmented='hot', save=False, path=None):
        """Plot the segmentation results.

        It is recommended to adjust the stardist parameters nms_thresh and prob_thresh based on this plot.

        Args:
            fig_size: Size of the figure.
            dpi: Dots per inch (DPI) of the figure.
            crop: If None, show the full image. Otherwise, crop should be
            cmap_segmented: Color map of the segmented image.
            save: If true, save the figure.
            path: Path to the save figure.
        """
        assert self.is_segmented, "Please conduct segmentation first."
        fig, ax = plt.subplots(1, 2, figsize=fig_size, dpi=dpi)
        if crop is None:
            img = self.img
            img_segmented = self.label
        else:
            img = self.img[crop[0]:crop[1], crop[2]:crop[3]]
            img_segmented = self.label[crop[0]:crop[1], crop[2]:crop[3]]
        ax[0].imshow(img)
        ax[0].set_title("Original image")
        ax[1].imshow(img_segmented, cmap=cmap_segmented)
        ax[1].set_title("Segmented image")
        ax[0].axis('off')
        ax[1].axis('off')
        if save:
            plt.savefig(path, bbox_inches='tight')
        plt.show()


def change_predict_defaults(predict_function):
    """Wrap prediction function for tensorflow>=2.9.0 to reduce outputs.

    Start from tensorflow 2.9.0, the default verbose value in function tensorflow.keras.Model.predict is set to 'auto',
    which takes value 1 for the most of the time. Thus, this function is a decorator to wrap the predict function
    and set the default verbose to 0.

    Args:
        predict_function: tensorflow.keras.Model.predict
    """
    def wrapper(instance, x, verbose=0, **kwargs):
        return predict_function(instance, x, verbose=verbose, **kwargs)
    return wrapper


def cell_boundary(nucleus_location, img_size, max_dist, max_area, search_direction, verbose=0, delta=1):
    """
    Infer the cell boundary.

    Args:
        nucleus_location: Coordinates of the nucleus centers.
        img_size: The size of the image.
        max_dist: The largest distance from the cell boundary to the nucleus center.
        max_area: Largest area of a cell. Segmented nuclei with size larger than this value are discarded.
        search_direction: The search direction when determine the cell boundary.
        verbose: Whether to print progress.
        delta: Increment of the radius in each round.

    Returns:
        Dictionary of the cell boundary information.
    """
    img_cell = np.zeros(img_size)
    n_nuclei = len(nucleus_location)
    n_pixel = [0] * n_nuclei
    nuclei_left = set(range(1, n_nuclei+1))

    radius = 0
    while nuclei_left and radius < max_dist:
        radius += delta
        if verbose:
            print('r: '+str(radius)+'\tThere are ' + str(len(nuclei_left)) + ' nuclei left.')
        pixel_check = []
        for dx in range(-radius-1, radius+2):
            for dy in range(-radius-1, radius+2):
                if (radius-delta)**2 <= dx**2+dy**2 <= radius**2:
                    pixel_check += [[dx, dy]]

        nuclei_remove = set()
        for i in nuclei_left:
            border_temp = []
            c_y = int(nucleus_location[i-1, 0])
            c_x = int(nucleus_location[i-1, 1])
            flag = True  # Whether the expended space are all occupied by background or other cells.
            for dx, dy in pixel_check:
                x1 = c_x + dx
                y1 = c_y + dy
                if 0 <= x1 <= img_cell.shape[0]-1 and 0 <= y1 <= img_cell.shape[1]-1 and dx**2+dy**2 <= radius**2:
                    if img_cell[x1, y1] == 0:
                        img_cell[x1, y1] = i
                        border_temp += [[x1, y1]]
                    if img_cell[x1, y1] == i:
                        flag = False
            n_pixel[i-1] += len(border_temp)
            if n_pixel[i-1] > max_area or (len(border_temp) == 0 and flag):
                nuclei_remove.add(i)
        for i in nuclei_remove:
            nuclei_left.remove(i)

    img_cell_boundaries = np.zeros((img_cell.shape[0], img_cell.shape[1], 3))
    individual_boundary = {i:[] for i in range(n_nuclei+1)}
    # for x in tqdm(range(len(img_cell))):
    #     for y in range(len(img_cell[0])):
    #         idx = img_cell[x, y]
    #         for k in range(len(search_direction)):
    #             x1, y1 = x + search_direction[k][0], y + search_direction[k][1]
    #             if 0 <= x1 <= img_cell.shape[0]-1 and 0 <= y1 <= img_cell.shape[1]-1 and img_cell[x1, y1] != idx:
    #                 img_cell_boundaries[x, y] = [150, 150, 150]
    #                 individual_boundary[idx] += [[x, y]]
    #                 break

    search_direction = np.array(search_direction)
    boundary_img = np.zeros_like(img_cell)
    for dx, dy in tqdm(search_direction):
        d = max([abs(dx), abs(dy)])
        kernel = np.zeros((2*d+1, 2*d+1))
        kernel[dy+d, dx+d] = 1
        kernel[d, d] = -1
        convolved = convolve2d(img_cell, kernel, mode='same')
        boundary_img += np.where(convolved != 0, 1, 0)
    img_cell_boundaries[boundary_img > 0] = [150, 150, 150]
    # for i in tqdm(range(n_nuclei+1)):
    #     y, x = np.where((img_cell == i) & (boundary_img > 0))
    #     individual_boundary[i] += list(zip(x, y))
    for x in tqdm(range(len(img_cell))):
        for y in range(len(img_cell[0])):
            if boundary_img[x, y] > 0:
                individual_boundary[img_cell[x, y]] += [[x, y]]

    img_cell_boundaries = img_cell_boundaries.astype(np.int32)
    individual_boundary = {i: np.array(individual_boundary[i]) for i in range(n_nuclei+1)}
    d = {'img_cell': img_cell, 'cell_boundary': img_cell_boundaries, 'size': n_pixel,
         'individual_boundary': individual_boundary}
    return d


def add_boundary(img, boundary):
    img = img.copy()
    assert img.shape[:2] == boundary.shape[:2]
    for i in range(len(img)):
        for j in range(len(img[0])):
            if boundary[i, j, 0] > 0:
                img[i, j] = [50, 50, 50]
    return img
