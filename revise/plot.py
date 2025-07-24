import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import cv2 as cv
import matplotlib.patches as patches
import seaborn as sns

def plot_sc_ref(sc_ref, type_list, fig_size=(10, 4), dpi=300):
    """
    Plot the heatmap of the single cell reference

    Args:
        sc_ref: scRNA reference. np.ndarray n_type*n_gene.
        type_list: List of the cell types.
        fig_size: Initial size of the figure.
        dpi: Dots per inch (DPI) of the figure.
    """
    fig_size_adjust = (fig_size[0], fig_size[1]*sc_ref.shape[0]/20)
    plt.figure(figsize=fig_size_adjust, dpi=dpi)
    sc_ref_df = pd.DataFrame(sc_ref, index=type_list)
    sns.heatmap(sc_ref_df, robust=True)
    plt.show()


def plot_heatmap(adata_sc, key_type, fig_size=(10, 4), dpi=300, save=False, out_dir=""):
    """
    Plot the heatmap of the mean expression.

    Args:
        adata_sc: scRNA data (Anndata).
        key_type: The key that is used to extract cell type information from adata_sc.obs.
        fig_size: Initial size of the figure.
        dpi: Dots per inch (DPI) of the figure.
        save: Whether to save the heatmap.
        out_dir: Output directory.
    """
    X = adata_sc.X if type(adata_sc.X) is np.ndarray else adata_sc.X.toarray()
    type_list = sorted(list(adata_sc.obs[key_type].unique()))  # list of the cell type.
    n_gene, n_type = adata_sc.shape[1], len(type_list)
    expression_mu = np.zeros((n_type, n_gene))  # Mean expression of each gene in each cell type.

    for i in range(n_type):
        data_temp = X[adata_sc.obs[key_type] == type_list[i]]
        expression_mu[i] = np.mean(data_temp, axis=0)
    del X

    expression_mu = expression_mu/np.max(expression_mu, axis=0)
    fig_size_adjust = (fig_size[0], fig_size[1]*n_type/20)
    plt.figure(figsize=fig_size_adjust, dpi=dpi)
    sc_ref_df = pd.DataFrame(expression_mu, index=type_list)
    sns.heatmap(sc_ref_df, robust=True)
    if save:
        plt.savefig(out_dir+'heatmap.jpg', bbox_inches='tight')
    plt.show()


class Plot_Visium:
    def __init__(self, segmentation, boundary_dict, type_list, colors=None):
        """
        Args:
            segmentation: Object of segmentation.Segmentation.
            boundary_dict: Output of function segmentation.cell_boundary.
            type_list: List of the cell types.
            colors: Colors of each cell type.
        """
        self.img = segmentation.img  # Original image.
        if np.max(self.img) <= 1:
            self.img = (self.img*255).astype(np.int32)
        self.img_seg = segmentation.label  # Segmented image.
        self.n_cell_df = segmentation.n_cell_df  # Dataframe shown the cells in each spot.
        self.nucleus_boundary = segmentation.nucleus_boundary.astype(np.int32)  # Segmented nucleus boundaries.
        self.nucleus_df = segmentation.nucleus_df
        self.spot_radius = segmentation.spot_radius
        self.spot_centers = segmentation.spot_center

        self.nucleus_centers = self.nucleus_df[['x', 'y']].values.astype(np.int32)  # Segmented nucleus centers.
        self.img_cell = boundary_dict['img_cell'].astype(np.int32)   # Inferred cell location.
        self.type_mask = None
        self.img_boundary = boundary_dict['cell_boundary']
        self.individual_boundary = boundary_dict['individual_boundary']
        self.img_size = self.img_seg.shape[:2]
        self.type_list = type_list

        if colors is None:
            self.colors = np.array(list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("tab20c").colors))
            self.colors = (self.colors*255).astype(np.int32)
        else:
            self.colors = colors
        self.color_dict = {type_: list(self.colors[i]) for i, type_ in enumerate(type_list)}

    def plot(self, background=False, cell='both', shape='cell', circle_size=10, boundary=None,
             save='Visium_plot.png', background_alpha=0.5, spot=True, spot_width=2, spot_color=(0, 0, 255),
             cell_boundary_color=(100, 100, 100), dpi=300):
        """
        Args:
            background: If show the background.
            cell: Which group of cell shapes to plot? [both, in, out]
            shape: Which shape to plot? [cell, nucleus, circle]
            circle_size: Size of the nuclei.
            boundary: Which group of cell boundary to plot? [both, in, out]
            save: If not none, save the figure to the path.
            background_alpha: Opacity of the background figure.
            spot: If plot the spot.
            spot_width: Width of the spot.
            spot_color: Color of the spot.
            cell_boundary_color: Color of the cell boundaries.
            dpi: DPI of the image plotted.
        """
        img = self.img.copy()*background_alpha if background else np.zeros(self.img.shape)
        img = img.astype(np.int32)

        group_dict = {'both': [True, False], 'in': [True], 'out': [False], None:[], 'None':[]}
        type_annotation = list(self.nucleus_df['cell_type'])
        in_spot = list(self.nucleus_df['in_spot'])

        print('Adding cells.')
        if shape == 'cell':
            d = {t: i for i, t in enumerate(self.type_list)}
            type_mask = np.zeros((img.shape[0], img.shape[1], len(self.type_list)), dtype=bool)
            for i in tqdm(range(len(self.img_cell))):
                for j in range(len(self.img_cell[0])):
                    k = self.img_cell[i, j]
                    if (0 < k <= len(type_annotation) and type_annotation[k-1] in d.keys()
                            and in_spot[k-1] in group_dict[cell]):
                        type_mask[i, j, d[type_annotation[k-1]]] = True
            for i, type_ in tqdm(enumerate(self.type_list)):
                color = np.array(self.color_dict[type_]).astype(np.int32)
                img[type_mask[:, :, i]] = color
            # img.astype(np.int32)

        for i, type_ in tqdm(enumerate(type_annotation)):
            if in_spot[i] in group_dict[cell] and type_ in self.color_dict.keys():
                color = [int(i) for i in self.color_dict[type_]]
                if shape == 'circle':
                    img = cv.circle(img, self.nucleus_centers[i], circle_size, color, -1)
                elif shape == 'nucleus':
                    boundary_temp = self.nucleus_boundary[i].reshape((-1, 1, 2))
                    img = cv.fillPoly(img, [boundary_temp], color=color)

        boundary_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        for i in tqdm(range(len(type_annotation))):
            if (in_spot[i] in group_dict[boundary] and len(self.individual_boundary[i+1]) > 0
                    and type_annotation[i]) in self.color_dict.keys():
                index_temp = np.array(self.individual_boundary[i+1])
                boundary_mask[index_temp[:, 0], index_temp[:, 1]] = True
        img[boundary_mask] = cell_boundary_color

        if spot:
            print('Adding spots.')
            for i in range(len(self.spot_centers)):
                img = cv.circle(img, self.spot_centers[i], int(self.spot_radius), spot_color, spot_width)
        if save is not None:
            print('Saving the image.')
            img1 = img[:, :, [2, 1, 0]]
            cv.imwrite(save, img1)
        plt.figure(dpi=dpi)
        plt.imshow(img)

    def plot_legend(self, save=None, dpi=300, background=(0, 0, 0), ncol=1, word_color=(255, 255, 255), fontsize=6,
                    horizontal_distance=380):
        nrow = len(self.type_list)//ncol
        if nrow*ncol < len(self.type_list):
            nrow += 1
        img = np.ones((nrow*60+40, ncol*horizontal_distance-30, 3)) * np.array(background)
        img = img.astype(np.int32)

        fig, ax = plt.subplots(dpi=dpi)
        plt.imshow(img)
        i = 0  # col index
        j = 0  # row index
        word_color = (word_color[0]/255, word_color[1]/255, word_color[2]/255)
        for k, v in self.color_dict.items():
            ax.add_patch(patches.Rectangle((i*horizontal_distance+20, j*60+20), 100, 45,
                                           facecolor=np.array(v)/255, edgecolor='none'))
            ax.text(i*horizontal_distance+145, j*60+43, k, va="center", ha="left", fontsize=fontsize, color=word_color)
            j += 1
            if j == nrow:
                j = 0
                i += 1
        ax.axis('off')
        if save is not None:
            plt.savefig(save, bbox_inches='tight', pad_inches=0)
        plt.show()


class Plot_Xenium:
    def __init__(self, Xenium_img, cell_boundaries, nucleus_boundaries, type_list, cell_type, nucleus_centers):
        """
        Args:
            Xenium_img: Image of Xenium
            cell_boundaries: Coordinates of the cell boundaries on the image.
            nucleus_boundaries: Coordinates of the nucleus boundaries on the image.
            type_list: List of the cell types.
            cell_type: Annotation.
            nucleus_centers: Centers of the nuclei.
        """
        self.img = Xenium_img
        self.type_list = type_list
        self.cell_type = cell_type
        self.n_cell = len(cell_type)
        self.nucleus_centers = nucleus_centers.astype(np.int32)

        cell_boundaries_temp = cell_boundaries.iloc[:, 1:3].values.astype(np.int32)
        idx = [[] for _ in range(self.n_cell)]
        for i in tqdm(range(len(cell_boundaries))):
            cell_idx = int(cell_boundaries.iloc[i, 0]-1)
            idx[cell_idx] += [i]
        self.cell_boundaries = [cell_boundaries_temp[np.array(idx[i])].reshape((-1, 1, 2)) if len(idx[i])>0
                                else [] for i in tqdm(range(self.n_cell))]

        nucleus_boundaries_temp = nucleus_boundaries.iloc[:, 1:3].values.astype(np.int32)
        idx = [[] for _ in range(self.n_cell)]
        for i in tqdm(range(len(nucleus_boundaries))):
            cell_idx = int(nucleus_boundaries.iloc[i, 0]-1)
            idx[cell_idx] += [i]
        self.nucleus_boundaries = [nucleus_boundaries_temp[np.array(idx[i])].reshape((-1, 1, 2)) if len(idx[i])>0
                                   else [] for i in tqdm(range(self.n_cell))]

        self.colors = np.array(list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("tab20c").colors))
        self.colors = (self.colors*255).astype(np.int32)
        self.color_dict = {type_: list(self.colors[i]) for i, type_ in enumerate(type_list)}

    def plot(self, background=False, shape='cell', save='Xenium_plot.png', cell_boundaries=False, background_alpha=0.8,
             circle_size=10, cell_boundary_color=(100, 100, 100), cell_boundary_thickness=2):
        """
        Args:
            background: If show the background.
            shape: Which shape to plot? [cell, nucleus, circle]
            save: If not none, save the figure to the path.
            cell_boundaries: If plot the cell boundaries.
            background_alpha: Opacity of the background figure.
            circle_size: Size of the circle.
            cell_boundary_color: Color of the cell boundaries.
            cell_boundary_thickness: Thickness of the cell boundaries.
        """
        img = self.img.copy()*background_alpha if background else np.zeros(self.img.shape)
        img = img.astype(np.int32)
        print('Adding cellls.')
        for i, type_ in tqdm(enumerate(self.cell_type)):
            if type_ in self.color_dict.keys():
                color = [int(i) for i in self.color_dict[type_]]
                if shape == 'cell':
                    if len(self.cell_boundaries[i]) > 0:
                        img = cv.fillPoly(img, [self.cell_boundaries[i]], color=color)
                elif shape == 'nucleus':
                    if len(self.nucleus_boundaries[i]) > 0:
                        img = cv.fillPoly(img, [self.nucleus_boundaries[i]], color=color)
                elif shape == 'circle':
                    img = cv.circle(img, self.nucleus_centers[i], circle_size, color, -1)
        if cell_boundaries:
            for i, type_ in tqdm(enumerate(self.cell_type)):
                if len(self.cell_boundaries[i]) > 0:
                    img = cv.polylines(img, [self.cell_boundaries[i]], color=cell_boundary_color,
                                       thickness=cell_boundary_thickness, isClosed=True)

        if save is not None:
            print('Saving the image.')
            img1 = img[:, :, [2, 1, 0]]
            cv.imwrite(save, img1)
        plt.imshow(img)