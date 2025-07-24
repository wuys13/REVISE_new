import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from matplotlib.ticker import MaxNLocator


class Evaluation:
    def __init__(self, proportion_truth, proportion_estimated_list, methods, out_dir="", cluster=None, type_list=None,
                 colors=None, coordinates=None, min_spot_distance=112):
        """
        Args:
            proportion_truth: Ground truth of the cell proportion.
            proportion_estimated_list: List of estimated proportion by each method.
            methods: List of methods names.
            out_dir: Output directory.
            cluster: Cluster label of each spot.
            type_list: List of cell types
        """
        self.proportion_truth = proportion_truth
        self.proportion_estimated_list = proportion_estimated_list
        self.methods = methods
        self.metric_dict = dict()  # Saved metric values.
        self.n_method = len(methods)
        self.coordinates = coordinates
        self.min_spot_distance = min_spot_distance
        self.cluster = cluster
        self.type_list = type_list
        self.n_type = proportion_truth.shape[1]

        self.function_map = {'Absolute error': self.absolute_error, 'Square error': self.square_error,
                             'Cosine similarity': self.cosine, 'Correlation': self.correlation,
                             'JSD': self.JSD, 'Fraction of cells correctly mapped': self.correct_fraction}
        self.metric_type_dict = {'Spot': {'Cosine similarity', 'Absolute error', 'Square error', 'JSD', 'Correlation',
                                          'Fraction of cells correctly mapped'},
                                 'Cell type': {'Cosine similarity', 'Absolute error', 'Square error', 'JSD',
                                               'Fraction of cells correctly mapped', 'Correlation'},
                                 'Individual': {'Absolute error', 'Square error'}}
        if colors is None:
            self.colors = ["#3c93c2", "#089099", "#7ccba2", "#fcde9c", "#f0746e", "#dc3977", "#7c1d6f"]
        else:
            self.colors = colors
        self.colors = self.colors[:len(self.methods)]

        if out_dir:
            self.out_dir = out_dir if out_dir[-1] == '/' else out_dir + '/'
        else:
            self.out_dir = ''
        if self.out_dir and not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        if not os.path.exists(out_dir + 'figures'):
            os.mkdir(out_dir + 'figures')
        assert len(proportion_estimated_list) == self.n_method

    @staticmethod
    def absolute_error(proportion_truth: np.ndarray, proportion_estimated: np.ndarray, metric_type='Spot'):
        assert proportion_truth.shape == proportion_estimated.shape
        error = np.abs(proportion_truth - proportion_estimated)
        if metric_type == 'Individual':
            return error
        elif metric_type == 'Spot':
            return np.sum(error, axis=1)
        elif metric_type == 'Cell type':
            return np.sum(error, axis=0)
        else:
            raise ValueError(f"Invalid metric type {metric_type}")

    @staticmethod
    def square_error(proportion_truth: np.ndarray, proportion_estimated: np.ndarray, metric_type='Spot'):
        assert proportion_truth.shape == proportion_estimated.shape
        error = (proportion_truth - proportion_estimated) ** 2
        if metric_type == 'Individual':
            return error
        elif metric_type == 'Spot':
            return np.sum(error, axis=1)
        elif metric_type == 'Cell type':
            return np.sum(error, axis=0)
        else:
            raise ValueError(f"Invalid metric type {metric_type}")

    @staticmethod
    def cosine(proportion_truth: np.ndarray, proportion_estimated: np.ndarray, metric_type='Spot'):
        assert proportion_truth.shape == proportion_estimated.shape
        if metric_type == 'Spot':
            cosine_similarity = np.sum(proportion_truth * proportion_estimated, axis=1) / \
                                np.linalg.norm(proportion_estimated, axis=1) / np.linalg.norm(proportion_truth, axis=1)
            return cosine_similarity
        elif metric_type == 'Cell type':
            cosine_similarity = np.sum(proportion_truth * proportion_estimated, axis=0) / \
                                np.linalg.norm(proportion_estimated, axis=0) / np.linalg.norm(proportion_truth, axis=0)
            return cosine_similarity
        else:
            raise ValueError(f"Invalid metric type {metric_type}")

    @staticmethod
    def correlation(proportion_truth: np.ndarray, proportion_estimated: np.ndarray, metric_type='Spot'):
        assert proportion_truth.shape == proportion_estimated.shape
        if metric_type == 'Cell type':
            proportion_truth_centered = proportion_truth - np.mean(proportion_truth, axis=0)
            proportion_estimated_centered = proportion_estimated - np.mean(proportion_estimated, axis=0)
            correlation_values = np.sum(proportion_truth_centered * proportion_estimated_centered, axis=0) / \
                                 np.sqrt(np.sum(proportion_truth_centered ** 2, axis=0) *
                                         np.sum(proportion_estimated_centered ** 2, axis=0))
            return correlation_values
        elif metric_type == 'Spot':
            proportion_truth_centered = proportion_truth - np.mean(proportion_truth, axis=1, keepdims=True)
            proportion_estimated_centered = proportion_estimated - np.mean(proportion_estimated, axis=1, keepdims=True)
            correlation_values = np.sum(proportion_truth_centered * proportion_estimated_centered, axis=1) / \
                                 np.sqrt(np.sum(proportion_truth_centered ** 2, axis=1) *
                                         np.sum(proportion_estimated_centered ** 2, axis=1))
            return correlation_values
        else:
            raise ValueError(f"Invalid metric type {metric_type}")

    @staticmethod
    def correct_fraction(proportion_truth: np.ndarray, proportion_estimated: np.ndarray, metric_type='Spot'):
        assert proportion_truth.shape == proportion_estimated.shape
        correct_proportion = np.minimum(proportion_truth, proportion_estimated)
        if metric_type == 'Cell type':
            return np.sum(correct_proportion, axis=0) / np.sum(proportion_truth, axis=0)
        elif metric_type == 'Spot':
            return np.sum(correct_proportion, axis=1)
        else:
            raise ValueError(f"Invalid metric type {metric_type}")

    @staticmethod
    def JSD(proportion_truth: np.ndarray, proportion_estimated: np.ndarray, metric_type='Spot'):
        """
        Jensen–Shannon divergence
        Args:
            proportion_truth: Ground truth of the cell proportion.
            proportion_estimated: Estimated proportion.
            metric_type: How the metric is calculated.
        """
        assert proportion_truth.shape == proportion_estimated.shape
        if metric_type == 'Spot':
            return jensenshannon(proportion_truth, proportion_estimated, axis=1)
        elif metric_type == 'Cell type':
            return jensenshannon(proportion_truth, proportion_estimated, axis=0)
        else:
            raise ValueError(f"Invalid metric type {metric_type}")

    def evaluate_metric(self, metric='Cosine similarity', metric_type='Spot', region=None):
        """
        Evaluate the proportions based on the metric.

        Args:
            metric: Name of the metric.
            metric_type: How the metric is calculated. 'Spot': metric is calculated for each spot; 'Cell type',
                metric is calculated for each cell type; 'Individual': metric is calculated for each individual
                proportion estimation.
            region: The region that is being evaluated.
        """
        assert metric in self.metric_type_dict[metric_type]
        metric_values = []
        func = self.function_map.get(metric)
        if region is None:
            select = np.array([True]*len(self.proportion_truth))
        elif isinstance(region, list):
            select = np.array([i in region for i in self.cluster])
        else:
            select = np.array([i == region for i in self.cluster])

        if not any(select):
            raise ValueError(f'Region {region} do/does not exist.')
        elif metric_type == 'Cell type' and np.sum(select) == 1:
            raise ValueError(f'Region {region} only contain(s) one spot.')
        for i in range(self.n_method):
            metric_values.append(func(self.proportion_truth[select], self.proportion_estimated_list[i][select],
                                      metric_type))
        self.metric_dict[metric+' '+metric_type] = metric_values
        return metric_values

    def plot_metric(self, save=False, region=None, metric='Cosine similarity', metric_type='Spot', cell_types=None,
                    suffix='', show=True):
        """Plot the box plot of each method based on the metric.

        Box number equals to the number of methods.

        Args:
            save: If true, save the figure.
            region: Regions of the tissue.
            metric: Name of the metric.
            metric_type: How the metric is calculated. 'Spot': metric is calculated for each spot; 'Cell type',
                metric is calculated for each cell type; 'Individual': metric is calculated for each individual
                proportion estimation.
            cell_types: If metric_type is 'Cell type' and cell_types is not None, then only plot the results
                corresponding to the cell_types.
            suffix: suffix of the save file.
            show: Whether to show the figure
        """
        assert metric_type == 'Spot' or metric_type == 'Cell type'
        assert metric in self.metric_type_dict[metric_type]
        if metric+' '+metric_type not in self.metric_dict.keys():
            self.evaluate_metric(metric=metric, metric_type=metric_type, region=region)
        region_name = ''
        if region is not None:
            region_name = '_' + '+'.join(region) if isinstance(region, list) else '_' + region

        if metric_type == 'Cell type' and cell_types is not None:
            idx = np.array([np.where(self.type_list == type_)[0][0] for type_ in cell_types])
            metric_values = self.metric_dict[metric+' '+metric_type]
            metric_values = [m[idx] for m in metric_values]
            metric_values = np.concatenate(metric_values)
            methods_name = np.repeat(self.methods, len(idx))
        else:
            metric_values = np.concatenate(self.metric_dict[metric+' '+metric_type])
            methods_name = np.repeat(self.methods, len(self.metric_dict[metric+' '+metric_type][0]))

        df = pd.DataFrame({metric: metric_values, 'Method': methods_name})
        if show:
            plt.figure(dpi=300)
            palette = self.colors if len(self.methods) <= len(self.colors) else 'Dark2'
            ax = sns.boxplot(data=df, y=metric, x='Method', showfliers=False, palette=palette)
            if metric_type == 'Spot':
                ax = sns.stripplot(data=df, y=metric, x='Method', ax=ax, jitter=0.2, palette=palette, size=1)
            else:
                ax = sns.stripplot(data=df, y=metric, x='Method', ax=ax, jitter=True, color='black', size=1.5)
            for patch in ax.patches:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, .8))
            ax.set(xlabel='')
            sns.despine(top=True, right=True)
            plt.gca().yaxis.grid(False)
            plt.gca().xaxis.grid(False)
            plt.gca().spines['left'].set_color('black')
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().tick_params(left=True, axis='y', colors='black')
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
            if save:
                plt.savefig(f'{self.out_dir}figures/{metric}_{metric_type}{region_name}{suffix}.jpg', dpi=500,
                            bbox_inches='tight')
            plt.show()
        return df

    def plot_metric_spot_type(self, save=False, metric='Absolute error'):
        """
        Similar to plot_metric_spot, but the figures are separated for each cell type.
        """
        """
        Plot the box plot of each method based on the metric. Each value in box plot represents a spot.
        Box number equals to the number of methods.
        """
        assert metric in self.general_metric_names
        plt.figure(dpi=300)
        if metric not in self.metric_dict.keys():
            self.evaluate_metric(metric=metric)
        sns.set_palette("Dark2")
        n_spot = len(self.proportion_truth)
        metric_values = np.vstack(self.metric_dict[metric])
        metric_values = metric_values.flatten('F')
        methods_name = np.repeat(self.methods, n_spot)
        methods_name = np.tile(methods_name, self.n_type)
        cell_type = np.repeat(self.type_list, n_spot * self.n_method)

        # print(np.shape(metric_values), np.shape(methods_name), np.shape(cell_type))
        df = pd.DataFrame({'metric': metric_values, 'Method': methods_name, 'Cell type': cell_type})

        for cell_type in self.type_list:
            ax = sns.boxplot(data=df[df['Cell type'] == cell_type], y='metric', x='Method', showfliers=False)
            for patch in ax.patches:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, .7))
            ax = sns.stripplot(data=df[df['Cell type'] == cell_type], y='metric', x='Method', ax=ax, jitter=0.2,
                               palette='Dark2', size=2)
            ax.set(xlabel='')
            if save:
                cell_type = "".join(x for x in cell_type if x.isalnum())
                plt.savefig(f'{self.out_dir}figures/{metric} {cell_type}.jpg', dpi=500, bbox_inches='tight')
            plt.show()
            del ax

    def plot_metric_all(self, save=False, metric="Absolute error", region=None):
        assert metric in self.general_metric_names
        plt.figure(figsize=(self.n_method * self.n_type / 4, 5), dpi=300)
        if metric not in self.metric_dict.keys():
            self.evaluate_metric(metric=metric)
        sns.set_palette("Dark2")
        region_name = ''
        if region is None:
            metric_values = np.vstack(self.metric_dict[metric])
            metric_values = metric_values.flatten('F')
            n_spot = len(self.proportion_truth)
        else:
            if isinstance(region, list):
                select = np.array([i in region for i in self.cluster])
                region_name = '_' + '+'.join(region)
            else:
                select = np.array([i == region for i in self.cluster])
                region_name = '_' + region
            if not any(select):
                raise ValueError('Region must exist.')
            metric_values = [x[select] for x in self.metric_dict[metric]]
            metric_values = np.vstack(metric_values)
            metric_values = metric_values.flatten('F')
            n_spot = np.sum(select)

        methods_name = np.repeat(self.methods, n_spot)
        methods_name = np.tile(methods_name, self.n_type)
        cell_type = np.repeat(self.type_list, n_spot * self.n_method)

        df = pd.DataFrame({metric: metric_values, 'Method': methods_name, 'Cell type': cell_type})
        ax = sns.boxplot(data=df, y=metric, hue='Method', x='Cell type', flierprops={"marker": "o"}, dodge=True,
                         linewidth=0.6, fliersize=0.5)
        # sns.violinplot(data=df, y=metric, hue='Method', x='Cell type')
        # ax = sns.catplot(data=df, y=metric, hue='Method', x='Cell type', kind='boxen')
        # for patch in ax.patches:
        #     r, g, b, a = patch.get_facecolor()
        #     patch.set_facecolor((r, g, b, .7))
        # ax = sns.stripplot(data=df, y=metric, x='Method', hue='Cell type', ax=ax, jitter=0.2, palette='Dark2', size=2)
        # ax.set(xlabel='')
        if save:
            plt.savefig(f'{self.out_dir}figures/{metric}{region_name}.tiff', dpi=800, bbox_inches='tight')
        plt.show()

