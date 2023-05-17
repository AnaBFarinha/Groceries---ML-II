import pandas as pd

def convert_to_dt(df: pd.DataFrame,
                   variable: str
                   ) -> None:
    """
    Converts a column containing dates to a Pandas datetime object.

    ----------
    Parameters:
     - df (pd.DataFrame): The input DataFrame containing the date column.
        variable (str): The name of the column containing the dates.

    ----------
    Returns:
     - None, but converts the specified column

    """
    # Convert column to datetime pandas object
    df[variable] = pd.to_datetime(df[variable])
    
    # The function doesn't return anything, it modifies the input DataFrame in-place
    return None
