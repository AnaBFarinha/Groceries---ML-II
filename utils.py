import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

def calc_age(df: pd.DataFrame, 
             date: datetime.date = datetime.date(2023, 3, 31)
             ) -> None :
    
    """
    Calculate age from date of birth.

    ----------
    Parameters:
     - df (pd.DataFrame): dataframe with column 
            containing the date of birth.

    ----------
    Returns:
     - None, but a column is added to the dataframe.

   """
    
    # Calculate age and round it down
    df['age']=(np.fix((datetime.datetime.now() - df['customer_birthdate']).dt.days/365))

    # Convert series to int
    df['age'] = df['age'].astype(int)


def bar_chart(x: np.ndarray, y: np.ndarray,
               x_name: str, y_name: str,
               title: str, 
               color: str = '#e0218a'
               ) -> None:
    
    """
    Create a bar chart using matplotlib with 
        the specified x-axis and y-axis data.

    ----------
    Parameters:
     - x (np.ndarray): A numpy array with the x-axis data.
     - y (np.ndarray): A numpy array with the y-axis data.
     - x_name (str): The name of the x-axis.
     - y_name (str): The name of the y-axis.
     - title (str): The title of the chart.
     - color (str): The color of the bars. Defaults to '#e0218a'.

    ----------
    Returns:
        None, but shows a plot
    """
    
    # Set the x-axis range and ticks
    x_axis = range(min(x), max(x) + 1, 1)
    plt.xticks(x_axis)

    # Set the chart title and axis labels
    plt.title(title,
              fontdict={'fontsize': 13})
    plt.xlabel(x_name, labelpad=10)
    plt.ylabel(y_name, ha='right')

    # Create a bar chart using matplotlib with unique counts as x-axis and frequency as y-axis
    plt.bar(x, y, color = color)

    # Show the plot
    plt.show()


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


def categorical_numerical(
        dataframe: pd.DataFrame,
        drop_categorical: list = [],
        drop_numerical: list = [])-> Tuple[
        pd.DataFrame, pd.DataFrame]:

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


def data_load(path: str
              ) -> Tuple[pd.DataFrame, None]:

    """
    Import dataset.

    ----------
    Parameters:
     - path (str): location and name of the file.

    ----------
    Returns:
     - pd.DataFrame.
     - None, but a message is shown.

   """
    
    if path[-5:] == '.xlsv':
        return pd.read_excel(path)
    
    elif path[-4:] == '.csv':
        return pd.read_csv(path)
    
    else:
        print('Import for this file type is unavailable.')


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


def education(customer_name: str) -> str:

    """
    Return level of education according 
        to the information written on the 
        name of each customer.

    ----------
    Parameters:
     - customer_name (str): name of a customer.

    ----------
    Returns:
     - (str): level of education.

   """
    
    # Define the possible education levels
    education_levels = ['Phd.','Msc.', 'Bsc.']
    
    # Check if each education level is in the customer name
    for level in education_levels:
        if level in customer_name:
            # Return the education level found in the name
            return level
    
    # If no education level is found, assume high-school education
    return 'HS'


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


def IQR_outliers(df: pd.DataFrame,
                  variables: list[str]
                  ) -> None:
    """
    Identify potential outliers using the interquartile
      range (IQR) method.

    ----------
    Parameters:
     - df (pd.DataFrame): The pandas dataframe to be
        analyzed.
     - variables (list): A list of column names in the
        dataframe to check for outliers.

    ----------
    Returns:
     - None, but prints the potential outliers for each
        variable along with the number of outliers.
    """

    # Calculate the IQR for each variable
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1

    # Identify potential outliers for each variable
    lower_bound = q1 - (3 * iqr)
    upper_bound = q3 + (3 * iqr)

    outliers = {}
    for var in variables:
        outliers[var] = df[(df[var] < lower_bound[var]) | (df[var] > upper_bound[var])][var]

    # Print the potential outliers for each variable
    print('-------------------------------------')
    print('          Potential Outliers         ')
    print('-------------------------------------')

    for var in outliers:
        print(var, ': Number of Outliers ->', len(outliers[var]))
        if len(outliers[var]) != 0:
            outliers[var] = np.unique(outliers[var])
            print('  Outliers: ',outliers[var])
        print()
