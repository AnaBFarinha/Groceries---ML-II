import pandas as pd
from typing import Tuple

def data_load(path: str
              ) -> Tuple[pd.DataFrame, None]:

    """
    Import dataset.

    ----------
    Parameters:
    - path (str): location and name of the file

    ----------
    Returns:
     - pd.DataFrame
     - None, but a message is shown

   """
    
    if path[-5:] == '.xlsv':
        return pd.read_excel(path)
    
    elif path[-4:] == '.csv':
        return pd.read_csv(path)
    
    else:
        print('Import for this file type is unavailable.')
