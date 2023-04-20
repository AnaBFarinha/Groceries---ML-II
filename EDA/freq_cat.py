import pandas as pd
import numpy as np

def frequency_categorical(dataframe: pd.DataFrame,
                           categorical: str) -> tuple:
    """
    Return the frequency distribution of a 
        categorical variable in a pandas dataframe.

    ----------
    Parameters:
        dataframe (pd.DataFrame): A pandas dataframe
        categorical: The name of the categorical 
            variable to analyze.

    ----------
    Returns:
        A tuple containing two numpy arrays: one with
          the unique values of the categorical variable,
          and one with the frequency counts for each unique value.
    """
    
    # Get frequency counts for each unique value of the categorical variable
    ind_per_cat = dataframe[categorical].value_counts()

    # Count the number of unique frequency counts
    unique, counts = np.unique(ind_per_cat.values, return_counts=True)

    return unique, counts
