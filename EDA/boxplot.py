import pandas as pd
import matplotlib.pyplot as plt

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
    plt.boxplot(df, vert=False, labels=labels)

    # Set the title of the plot
    plt.title('Boxplot of ' + title)

    # Display the plot
    plt.show()
