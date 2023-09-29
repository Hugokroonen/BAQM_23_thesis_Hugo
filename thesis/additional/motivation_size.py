"""
Description:
    In this script, we calculate, for each motivation, in what % of all baskets this motivation is most important. 
"""

import pandas as pd

# Load the counts_phi dataset into a pandas DataFrame
counts_phi_df = pd.read_csv("output/M30/CTM/counts_basket.csv", header=None)

# Assign motivation indices 0, 1, 2, ... based on their location
counts_phi_df.columns = range(len(counts_phi_df.columns))

# Initialize a dictionary to keep count of how many times each motivation has the highest probability for a basket 
column_number_counts = {}

# Loop through each row and find the column (motivation) number with the highest number in that row
for index, row in counts_phi_df.iterrows():
    highest_value = row.max()
    column_number_with_highest = row.idxmax()  # Get the column number (index) with the highest value

    # Update the column_number_counts dictionary
    column_number_counts[column_number_with_highest] = column_number_counts.get(column_number_with_highest, 0) + 1

# Calculate the total number of rows in the DataFrame
total_rows = len(counts_phi_df)

# Calculate and print the summary statistics for each motivation - in what percentage of all baskets each motivation is most likely
print("Summary Statistics:")
for column_number, count in column_number_counts.items():
    percentage_with_highest = (count / total_rows) * 100
    print(f"Motivation {column_number} is most likely in {count} baskets ({percentage_with_highest:.2f}%).")
