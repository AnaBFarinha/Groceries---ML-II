import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap


def bar_chart(x: np.ndarray, y: np.ndarray,
               x_name: str, y_name: str,
               title: str, 
               color: str = '#e0218a'
               ) -> None:
    
    """
    Create a bar chart using matplotlib with 
        the specified x-axis and y-axis data.

    ----------
    Parameters:
     - x (np.ndarray): A numpy array with the x-axis data.
     - y (np.ndarray): A numpy array with the y-axis data.
     - x_name (str): The name of the x-axis.
     - y_name (str): The name of the y-axis.
     - title (str): The title of the chart.
     - color (str): The color of the bars. Defaults to '#e0218a'.

    ----------
    Returns:
        None, but shows a plot
    """
    
    # Set the x-axis range and ticks
    x_axis = range(min(x), max(x) + 1, 1)
    plt.xticks(x_axis)

    # Set the chart title and axis labels
    plt.title(title,
              fontdict={'fontsize': 13})
    plt.xlabel(x_name, labelpad=10)
    plt.ylabel(y_name, ha='right')

    # Create a bar chart using matplotlib with unique counts as x-axis and frequency as y-axis
    plt.bar(x, y, color = color)

    # Show the plot
    plt.show()


def boxplot(df: pd.DataFrame, labels: list,
             title: str) -> None:
    
    """
    Generate a box plot of the columns in a DataFrame.

    ----------
    Parameters:
     - df (pd.DataFrame): The input DataFrame 
        containing the data to plot.
     - labels (list): A list of strings to used
       as the labels for the box plot.
     - title (str): The title to display at the
       top of the plot.
    
    ----------
    Returns:
     - None, but a plot is shown.

    """
    
    # Generate a box plot of the data in the DataFrame
    plt.boxplot(df, vert=False, labels=labels, medianprops = {'color':'#e0218a'})

    # Set the title of the plot
    plt.title('Boxplot of ' + title)

    # Display the plot
    plt.show()


def plot_dendrogram(model: AgglomerativeClustering,
                     **kwargs) -> None:
    '''
    Create linkage matrix and then plot the dendrogram
    
    ----------
    Parameters:
     - model(HierarchicalClustering Model): hierarchical clustering model.
     - **kwargs

    ----------
    Returns:
     - None, but dendrogram plot is produced.
    '''
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def boxplot_color(df, variable, color_dict, clusters, xlabel, ylabel, title):
    # Create the figure and axes
    fig, ax = plt.subplots()

    # Create the boxplot
    boxplot = ax.boxplot([df[df['cluster_kmeans'] == cluster][variable].values for cluster in clusters],
                         patch_artist=True, medianprops={'color': '#000000'})

    # Set the facecolor for each box based on the cluster color
    for i, box in enumerate(boxplot['boxes']):
        cluster = clusters[i]
        color = color_dict.get(cluster)
        box.set(facecolor=color)

    # Customize the plot
    ax.set_xticklabels(range(0, len(clusters)))  # Set x-axis tick labels as cluster numbers
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Display the plot
    plt.show()



def visualize_dimensionality_reduction(transformation, targets):
    color_palette = ["#8B0000", "#FFE66D", "#FF5900", "#00B4D8", "#32CD32", "#C9A0DC", "#e0218a"]
    cmap = ListedColormap(color_palette)
    
    # Create a scatter plot
    plt.scatter(transformation[:, 0], transformation[:, 1], 
                c=np.array(targets).astype(int),
                cmap=cmap)
    
    labels = np.unique(targets)
    
    # Create a legend with the class labels and colors
    handles = [plt.scatter([],[], c=color_palette[i], label=label) for i, label in enumerate(labels)]
    plt.legend(handles=handles, title='Cluster')
    
    # Display the plot
    plt.show()



def map_clusters(df, color_dict):
    # Create a scatter mapbox figure
    fig = go.Figure()

    # Add scatter mapbox traces for each cluster
    for cluster, color in color_dict.items():
        filtered_df = df[df['cluster_kmeans'] == cluster]
        scatter = go.Scattermapbox(
            lat=filtered_df['latitude'],
            lon=filtered_df['longitude'],
            marker=dict(color=color),
            name=f'Cluster {cluster}',
            visible=True
        )
        fig.add_trace(scatter)

    # Set the mapbox style and center on Lisbon, change the color of the title
    fig.update_layout(
        mapbox_style='open-street-map',
        mapbox_center={'lat': 38.736946, 'lon': -9.142685},
        mapbox_zoom=9,
        title='Customers Addresses by Cluster',
        title_font_color="#e0218a"
    )

    # Create a list of checkbox options
    checkboxes = []
    for cluster, color in color_dict.items():
        checkbox = dict(
            label=f'Cluster {cluster}',
            method='update',
            args=[{'visible': [False if trace.name != f"Cluster {cluster}" else True for trace in fig.data]}]
        )
        checkboxes.append(checkbox)

    # Add the checkbox buttons
    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                buttons=list([
                    dict(
                        label='All',
                        method='update',
                        args=[{'visible': [True] * len(fig.data)}]
                    )
                ] + checkboxes),
                showactive=True,
            )
        ]
    )

    # Show the figure
    fig.show()