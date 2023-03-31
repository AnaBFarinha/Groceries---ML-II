import pandas as pd

def education(customer_name):
    # Define the possible education levels
    education_levels = ['Phd.','Msc.', 'Bsc.']
    
    # Check if each education level is in the customer name
    for level in education_levels:
        if level in customer_name:
            # Return the education level found in the name
            return level
    
    # If no education level is found, assume high-school education
    return 'High-school'
