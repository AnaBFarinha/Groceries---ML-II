import datetime
import numpy as np
import pandas as pd

def calc_age(df: pd.DataFrame, 
             date: datetime.date = datetime.date(2023, 3, 31)
             ) -> None :
    """
    Calculate age from date of birth.

    ----------
    Parameters:
    - df (pd.DataFrame): dataframe with column 
            containing the date of birth

    ----------
    Returns:
     - None, but a column is added to the dataframe

   """
    
    # Transform column to datetime pandas object
    df['customer_birthdate'] = pd.to_datetime(df.customer_birthdate)

    # Calculate age and round it down
    df['age']=(np.fix((datetime.datetime.now() - df['customer_birthdate']).dt.days/365))

    # Convert series to int
    (df['age']).astype(int)
