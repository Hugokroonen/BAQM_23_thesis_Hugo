"""
Description:
    In this script, we calculate the number of products needed per motivation to account for at least 50% of the probability mass.
"""

import pandas as pd

counts_phi_df = pd.read_csv('output/M30/CTM/counts_phi.csv', header=None)

# Function to find the number of products needed to exceed 50% of the total probability mass for a specific motivation
def find_rows_for_50_percent_sum(column_data):
    sorted_data = column_data.sort_values(ascending=False)
    total_sum = column_data.sum()
    current_sum = 0
    rows_needed = 0

    for value in sorted_data:
        current_sum += value
        rows_needed += 1
        if current_sum > total_sum * 0.5:
            break

    return rows_needed

# Loop through each motivation and calculate the number of products needed for each motivation 
for col in counts_phi_df.columns:
    column_data = counts_phi_df[col]
    rows_needed = find_rows_for_50_percent_sum(column_data)
    print(f"For motivation '{col}', you need {rows_needed} products to get more than 50% of the total probability mass.")