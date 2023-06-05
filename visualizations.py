import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap
from typing import Dict, List


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
    
    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])  
    # Total number of samples in the model
    n_samples = len(model.labels_)

    # Iterate through each merge in the hierarchical clustering model
    for i, merge in enumerate(model.children_):
        # Initialize the count for the current node
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                # Leaf node, increment the count
                current_count += 1
            else:
                # Non-leaf node, add the count from its children
                current_count += counts[child_idx - n_samples]
        # Store the count for the current node in the counts array
        counts[i] = current_count

    # Create the linkage matrix by horizontally stacking the 
    #   children indices, distances, and counts columns
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram using the linkage 
    #   matrix and any additional keyword arguments
    dendrogram(linkage_matrix, **kwargs)


def boxplot_color(df: pd.DataFrame, variable: str,
                   color_dict: Dict[int, str],
                   clusters: List[int],
                   xlabel: str, ylabel: str,
                   title: str) -> None:
    
    '''
    Create a boxplot to compare a variable across different clusters.

    ----------
    Parameters:
     - df (pd.DataFrame): The DataFrame containing the data.
     - variable (str): The variable to compare across clusters.
     - color_dict (Dict[int, str]): A dictionary mapping cluster 
            numbers to color codes.
     - clusters (List[int]): The list of cluster numbers to 
        include in the boxplot.
     - xlabel (str): The label for the x-axis.
     - ylabel (str): The label for the y-axis.
     - title (str): The title of the plot.

    ----------
    Returns:
     - None, but it shows a plot.

    '''

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


def visualize_dimensionality_reduction(transformation: np.ndarray,
                                        targets: List[str]) -> None:
    
    '''
    Visualize dimensionality reduction results using a scatter plot.

    ----------
    Parameters:
     - transformation (np.ndarray): The transformed data points
         with reduced dimensions.
     - targets (List[str]): The target labels for each data point.

    ----------
    Returns:
     - None, but displays a plot.

    '''

    # Define the color palette for the clusters
    color_palette = ["#8B0000", "#FFE66D", "#FF5900", "#00B4D8", "#32CD32", "#C9A0DC", "#e0218a"]
    cmap = ListedColormap(color_palette)
    
    # Create a scatter plot
    plt.scatter(transformation[:, 0], transformation[:, 1], 
                c=np.array(targets).astype(int),
                cmap=cmap)
    
    # Get names of the clusters
    labels = np.unique(targets)
    
    # Create a legend with the clusters and colors
    handles = [plt.scatter([],[], c=color_palette[i], label=label) for i, label in enumerate(labels)]
    plt.legend(handles=handles, title='Cluster')
    
    # Display the plot
    plt.show()


def map_clusters(df: pd.DataFrame,
                  color_dict: Dict[int, str]
                  ) -> None:
    '''
    Create a scatter mapbox plot to visualize 
        customers' addresses by cluster.

    ----------
    Parameters:
     - df (pd.DataFrame): The DataFrame containing the data,
         including 'latitude', 'longitude', and 
         'cluster_kmeans' columns.
     - color_dict (Dict[int, str]): A dictionary mapping 
        cluster numbers to color codes.
    
    ----------
    Returns:
     - None, but displays an interative map.

    '''

    # Create a scatter mapbox figure
    fig = go.Figure()

    # Add scatter mapbox traces for each cluster
    for cluster, color in color_dict.items():
        # Filter the DataFrame based on the current cluster
        filtered_df = df[df['cluster_kmeans'] == cluster]
        # Create a scatter mapbox trace for the current cluster
        scatter = go.Scattermapbox(
            lat=filtered_df['latitude'],
            lon=filtered_df['longitude'],
            marker=dict(color=color),
            name=f'Cluster {cluster}',
            visible=True
        )
        # Add the trace to the figure
        fig.add_trace(scatter)

    # Set the mapbox style and center on Lisbon, change the color of the title
    fig.update_layout(
        mapbox_style='open-street-map',
        mapbox_center={'lat': 38.736946, 'lon': -9.142685},
        mapbox_zoom=9,
        title='Customers Addresses by Cluster',
        title_font_color="#e0218a"
    )

    # Create a list of checkbox options for cluster visibility control
    checkboxes = []
    for cluster, color in color_dict.items():
        # Create a checkbox option for the current cluster
        checkbox = dict(
            label=f'Cluster {cluster}',
            method='update',
            args=[{'visible': [False if trace.name != f"Cluster {cluster}" else True for trace in fig.data]}]
        )
        checkboxes.append(checkbox)

    # Add the checkbox buttons to the map
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


def plot_AR(rules_grocery: pd.DataFrame) -> None:
    
    """
    Plot association rules as a scatter plot.

    ----------
    Parameters:
     - rules_grocery (pd.DataFrame): DataFrame containing
         association rules.

    ----------
    Returns:
     - None, but it shows an interactive plot.
    """

    # Convert the 'antecedents' and 'consequents' columns from frozenset to list
    rules_grocery['antecedents'] = rules_grocery['antecedents'].apply(list)
    rules_grocery['consequents'] = rules_grocery['consequents'].apply(list)

    # Convert the DataFrame to a JSON-compatible format
    data = rules_grocery.to_dict(orient='records')

    # Create a scatter trace
    scatter_trace = go.Scatter(
        x=[rule['lift'] for rule in data],
        y=[rule['confidence'] for rule in data],
        mode='markers',
        text=[f"Antecedents: {rule['antecedents']}<br>Consequents: {rule['consequents']}<br>Support: {rule['support']}"
            for rule in data],
        hovertemplate='<b>Rule Details:</b><br>%{text}<br>'
                    '<b>Lift:</b> %{x}<br>'
                    '<b>Confidence:</b> %{y}<extra></extra>',
        customdata=data,  # Store the entire DataFrame as custom data
        marker=dict(
            color='#e0218a'  # Set the color of the points (e.g., 'blue')
        )
    )

    # Create the layout
    layout = go.Layout(
        title='Association Rules',
        xaxis=dict(title='Lift'),
        yaxis=dict(title='Confidence'),
        hovermode='closest',
        plot_bgcolor='white'
    )

    # Create the figure
    fig = go.Figure(data=[scatter_trace], layout=layout)

    # Change grid color
    fig = fig.update_yaxes(gridcolor='lightgrey')
    fig = fig.update_xaxes(gridcolor='lightgrey')

    # Define the callback function for clicking on a point
    def point_click_callback(trace, points, state):
        rule = points.point.customdata[points.point.point_inds[0]]
        print(rule)  # Modify this to perform the desired action with the rule details

    # Assign the callback function to the scatter trace
    scatter_trace.on_click(point_click_callback)

    # Show the figure
    fig.show()