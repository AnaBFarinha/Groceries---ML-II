import pandas as pd
import numpy as np

def float_to_int(df: pd.DataFrame,
                  variables: list[str]
                  ) -> pd.DataFrame:
    
    """
    Converts specified columns in a pandas dataframe 
        from float to integer data type.

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

    # Iterate through each variable and convert from float to integer
    for i in variables:

        # Check if column contains any missing values
        if df[i].isnull().values.any():
            continue
        
        else:
            df[i] = df[i].astype(int)

    # Return the modified dataframe
    return df
