#Used to visualize the AD progression trajectory in the latent space

import pandas as pd
import matplotlib.pyplot as plt


def read_tsne_data(csv_file):
    tsne_df = pd.read_csv(csv_file)
    return tsne_df[['tSNE1', 'tSNE2', 'tSNE3']].values, tsne_df['Label'].values


def plot_3d_tsne(data, labels, title):
    z_tsne = data

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')


    colors = {
        0: 'royalblue',
        1: 'lightskyblue',
        2: 'green',
        3: 'pink',
        4: 'darkred'
    }


    label_order = [3, 1, 4, 0, 2]

    for label in label_order:

        label_indices = labels == label
        x = z_tsne[label_indices, 0]
        y = z_tsne[label_indices, 1]
        z = z_tsne[label_indices, 2]

        ax.scatter(x, y, z, color=colors[label], label=f"Label {label}", alpha=0.8, s=5)

    ax.set_xlabel('tSNE1', fontsize=12, labelpad=10)
    ax.set_ylabel('tSNE2', fontsize=12, labelpad=10)
    ax.set_zlabel('tSNE3', fontsize=12, labelpad=10)

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    ax.set_box_aspect([1, 1, 1])

    ax.xaxis.set_tick_params(direction='out', width=1.5, pad=5)
    ax.yaxis.set_tick_params(direction='out', width=1.5, pad=5)
    ax.zaxis.set_tick_params(direction='out', width=1.5, pad=5)

    axis_limit = 23
    ax.set_xlim([-axis_limit, axis_limit])
    ax.set_ylim([-axis_limit, axis_limit])
    ax.set_zlim([-axis_limit, axis_limit])

    #ax.view_init(elev=40, azim=-290)

    ax.legend(title="Labels", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=20)

    ax.set_title(title, fontsize=14)

    plt.savefig(f't-SNE_visualization.png', bbox_inches='tight', dpi=600)


    # plt.show()



csv_file = f'model/tSNE_results_with_labels.csv'
tsne_data, labels = read_tsne_data(csv_file)

plot_3d_tsne(tsne_data, labels, 't-SNE Visualization from CSV')
