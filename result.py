import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from scipy.interpolate import griddata


def analyze_results(k_max=70, embedding_max=110):
    results = {}

    for name in ['org_cls_report', 'indirect_concrete_cls_report', 'concrete_cls_report']:
        k_values = range(10, k_max, 10)
        embedding_values = range(10, embedding_max, 10)
        acc_data = pd.DataFrame(index=k_values, columns=embedding_values, dtype=float)
        macro_precision_data = pd.DataFrame(index=k_values, columns=embedding_values, dtype=float)

        for k in k_values:
            for embedding in embedding_values:
                dir_name = f'result_cls_pancreas-k_{k}-h_256-e_{embedding}-ep_200-random_42'
                try:
                    df = pd.read_csv(f'{dir_name}/{name}.csv', index_col=0)
                    acc_data.at[k, embedding] = df.iloc[-3, 0]  # Assuming the accuracy is in the third last row, first column
                    macro_precision_data.at[k, embedding] = df.iloc[-2, 0]  # Assuming the macro precision is in the second last row, first column
                except FileNotFoundError:
                    print(f"File not found: {dir_name}/{name}.csv")
                    # If file is not found, we skip this data point and leave it NaN

        results[name] = {'accuracy': acc_data, 'macro_precision': macro_precision_data}

    return results


def plot_k_embedding(results, metric='accuracy'):
    for idx, (name, metrics) in enumerate(results.items()):
        data = metrics[metric].copy()
        k_values = [float(k) for k in data.index]
        embedding_values = [float(e) for e in data.columns]

        # Heatmap
        plt.figure(figsize=(7, 7))
        sns.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': metric}, fmt=".3f")
        plt.title(f'{name} - {metric} Heatmap')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('k Value')
        plt.tight_layout()  # Adjust layout to make sure everything fits without overlapping
        plt.show()

        # 3D Surface Plot
        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(embedding_values, k_values)
        Z = data.values

        surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(data.min().min(), data.max().max())
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5, label=metric)

        # Set labels
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('k Value')
        ax.set_zlabel(metric)
        ax.set_title(f'{name} - {metric} 3D Surface Plot')

        plt.tight_layout()
        plt.show()


def plot_k_embedding_contour(results, metric='accuracy'):
    for idx, (name, metrics) in enumerate(results.items()):
        data = metrics[metric].copy()
        k_values = [float(k) for k in data.index]
        embedding_values = [float(e) for e in data.columns]

        # Prepare data for contour plotting
        X, Y = np.meshgrid(embedding_values, k_values)
        Z = data.values

        # Find the top 3 maximum values and their positions
        flat_indices = np.argpartition(Z.flatten(), -3)[-3:]  # Get indices of top 3 values
        max_vals_and_positions = [(Z.flatten()[i], np.unravel_index(i, Z.shape)) for i in flat_indices]
        max_vals_and_positions.sort(reverse=True, key=lambda x: x[0])  # Sort by value in descending order

        # Contour Plot
        plt.figure(figsize=(14, 7))
        contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(contour, label=metric)  # Add color bar
        plt.title(f'{name} - {metric} Contour Plot')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('k Value')

        # Annotate the top 3 maximum values on the plot
        for val, (i, j) in max_vals_and_positions:
            max_k = k_values[i]
            max_embedding = embedding_values[j]
            plt.scatter(max_embedding, max_k, color='black', zorder=10)  # Highlight the point
            plt.text(max_embedding, max_k, f'\n({max_embedding}, {max_k})\n{val:.3f}', 
                     horizontalalignment='center', verticalalignment='top', 
                     color='black', fontsize=12, fontweight='bold', zorder=10)

        plt.tight_layout()  # Adjust layout to make sure everything fits without overlapping
        plt.savefig(f'{name}_{metric}_contour.png')
        plt.close()
        # Optionally, you can also add contour lines on top of the filled contours
        # contour_lines = plt.contour(contour, colors='black')
        # plt.clabel(contour_lines, inline=True, fontsize=8)


if __name__ == '__main__':
    # Analyze and get the results
    dfs = analyze_results()

    # Plot heatmaps and 3D surface plots for both accuracy and macro_precision
    # plot_k_embedding(dfs, metric='accuracy')
    plot_k_embedding_contour(dfs, metric='accuracy')
    plot_k_embedding_contour(dfs, metric='macro_precision')