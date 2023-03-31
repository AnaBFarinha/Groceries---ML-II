import pandas as pd

def education(customer_name: str) -> str:

    """
    Return level of education according 
        to the information written on the 
        name of each customer

    ----------
    Parameters:
    - customer_name (str): name of a customer

    ----------
    Returns:
     - (str): level of education

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
