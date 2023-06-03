import numpy as np
import matplotlib as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram


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


def plot_dendrogram(model, **kwargs):
    '''
    Create linkage matrix and then plot the dendrogram
    Arguments: 
    - model(HierarchicalClustering Model): hierarchical clustering model.
    - **kwargs
    Returns:
    None, but dendrogram plot is produced.
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