"""
Description:
    In this script, we write the correlation matrix to a separate file and analyse the correlations.
"""

import numpy as np
import pandas as pd

# Read raw lambda_kappa output 
precision_matrix = pd.read_csv("output/M30/CTM/lambda_kappa.csv")

# Take the inverse of the precision matrix to obtain the correlation matrix 
correlation_matrix = np.linalg.inv(precision_matrix)

# Set diagonal elements to 0, which is helpful for further analysis (actual values are 1)
np.fill_diagonal(correlation_matrix, 0)

# Create pandas dataframe 
correlation_df = pd.DataFrame(correlation_matrix)

# Replace 'path/to/your/file.xlsx' with the desired file path and name with the '.xlsx' extension
output_file_path = 'output/M30/CTM/correlation_matrix.xlsx'

# Write the DataFrame to an Excel file
correlation_df.to_excel(output_file_path, index=False, header=False)

# Get the number of columns in the correlation matrix
num_columns = correlation_matrix.shape[1]

# Analysis

# Loop through each column and find the row index with the highest value
for col_index in range(num_columns):
    max_value_row_index = np.argmax(correlation_matrix[:, col_index])
    max_value = correlation_matrix[max_value_row_index, col_index]
    print(f"For motivation {col_index}, the motivation with the highest correlation is motivation {max_value_row_index}, with a correlation of {max_value}.")

# Loop through each column and find the row index with the lowest value
for col_index in range(num_columns):
    max_value_row_index = np.argmin(correlation_matrix[:, col_index])
    max_value = correlation_matrix[max_value_row_index, col_index]
    print(f"For motivation {col_index}, the motivation with the lowest correlation is motivation {max_value_row_index}, with a correlation of {max_value}.")

