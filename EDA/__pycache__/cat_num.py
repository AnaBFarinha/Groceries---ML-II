import pandas as pd
from typing import Tuple

def categorical_numerical(
        dataframe: pd.DataFrame,
        drop_categorical: list = [],
        drop_numerical: list = [])-> Tuple(
    pd.DataFrame, pd.DataFrame
    ):

    """
    Separate a pandas dataframe into categorical
      and numerical dataframes, with optional columns to drop.

    ----------
    Parameters:
     - dataframe (pd.DataFrame): A pandas dataframe to separate.
     - drop_categorical (list): A list of column names to drop from
        the categorical dataframe.
     - drop_numerical (list): A list of column names to drop from
        the numerical dataframe.

    ----------
    Returns:
     - Tuple(pd.DataFrame, pd.DataFrame): one with categorical 
        variables and another with numerical variables.
    """

     # Separate categorical and numerical variables
    categorical_df = dataframe.loc[:,(dataframe.dtypes == 'object') == True]
    numerical_df = dataframe.loc[:,(dataframe.dtypes != 'object') == True]

    # Drop specified columns from each dataframe
    categorical_df = categorical_df.drop(drop_categorical, axis=1)
    numerical_df = numerical_df.drop(drop_numerical, axis=1)

    return categorical_df, numerical_df