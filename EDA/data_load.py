import pandas as pd

def data_load(path: str):
            
    if path[-5:] == '.xlsv':
        return pd.read_excel(path)
    if path[-4:] == '.csv':
        return pd.read_csv(path)
    else:
        print('Import for this file type is unavailable.')
