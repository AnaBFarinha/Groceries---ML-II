import pandas as pd

def cat_to_binary(dataframe: pd.DataFrame,
                   cat_name: str,
                   new_name: str) -> pd.DataFrame:
    
    """
    Converts a categorical variable in a Pandas 
    DataFrame to a binary variable and adds it to
    the DataFrame.
    
    ----------
    Parameters:
     - dataframe (pd.DataFrame): The input 
        DataFrame containing the categorical
        variable.
     - cat_name (str): The name of the categorical
        variable to be converted.
     - new_name (str): The name of the new 
        binary variable to be created.

    ---------- 
    Returns:
     - pd.DataFrame: The modified DataFrame 
        with the new binary variable added.
    """
    
    # Create a new DataFrame with the categorical 
    # variable transformed into the binary variable
    new_df = pd.get_dummies(dataframe[cat_name])[new_name]
    
    # Concatenate the new binary column with the 
    # original DataFrame and drop the original categorical column
    dataframe = pd.concat([dataframe.drop(cat_name, axis=1), new_df], axis=1)
    
    # Return the modified DataFrame with the 
    # new binary variable added
    return dataframe
