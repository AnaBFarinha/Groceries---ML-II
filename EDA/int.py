import pandas as pd
import numpy as np

def float_to_int(df: pd.DataFrame,
                  variables: list[str]
                  ) -> pd.DataFrame:
    """
    Converts specified columns in a pandas dataframe
      from float to integer data type, accounting 
      for missing values.

    ----------
    Parameters:
     - df (pd.DataFrame): The pandas dataframe 
        to be modified.
     - variables (list[str]): A list of column names
         in the dataframe to be converted.

    ----------
    Returns:
     - df (pd.DataFrame): The modified pandas dataframe
         with specified columns converted to integer data type.

    """

    # Iterate through each variable in the list of specified columns
    for variable in variables:
        # Check if the variable is of float data type
        if df[variable].dtype == 'float64':
            # Replace missing values with np.nan and convert to 
            # nullable integer data type 'Int64'
            df[variable] = df[variable].fillna(np.nan).astype('Int64')

    # Return the modified dataframe with specified columns converted
    # to integer data type
    return df

