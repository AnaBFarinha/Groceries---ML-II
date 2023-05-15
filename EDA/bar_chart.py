import matplotlib.pyplot as plt
import numpy as np

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

