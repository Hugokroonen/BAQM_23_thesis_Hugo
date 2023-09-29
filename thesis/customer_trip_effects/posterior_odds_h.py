"""
Description:
    In this script, we calculate the posterior odds ratios for the customer-specific variables. We also analyse the resulting
    customer-specific effects. 
"""

import numpy as np
import pandas as pd

# Load estimated gamma data
gamma_data = pd.read_csv("output/M30/CTM/gamma.csv")

# Load estimated beta data
beta_data = pd.read_csv("output/M30/CTM/beta.csv")

# Load estimated kappa data
kappa_data = pd.read_csv("output/M30/CTM/kappa.csv")

# Load basket data 
xib_data = pd.read_csv("data/x.csv")

# Load customer data 
hi_data = pd.read_csv("data/h.csv")

# Obtain the population-level mean of motivation intercepts kappa
kappa_mean = np.mean(kappa_data, axis=0)

# Function to calculate the odds ratio
def calculate_odds_ratio(kappa_mean, beta_data, gamma_data, xib_data, hi_data, m, variable_index):

    # Calculate the baseline alpha for all motivations
    alpha_baseline = []
    sum_alpha_baseline = 0

    # For all motivations
    # Get kappa mean for this motivation, and use x and w mean (average shopping trip) beta and gamma coefficients corresponding 
    # to the motivation 

    for i in range(len(gamma_data)):
        alpha_i = np.exp(kappa_mean[i] + np.dot(xib_data.mean(axis=0), beta_data.iloc[i]) + np.dot(hi_data.mean(axis=0), gamma_data.iloc[i]))
        alpha_baseline.append(alpha_i)
        sum_alpha_baseline += alpha_i

    wi_changed = np.copy(hi_data) 
    
    # Set customer characteristic (w) to one specific value

    # Variables: Unknown age, age0-25, age 25-35,  age 35-45, age 55-65, age 65+, city 
    if variable_index in [0, 1, 2, 3, 4, 5, 6]:
        wi_changed[:, variable_index] = 1
    # Set all ages variables to 0, to get baseline age column 
    elif variable_index == 7:
        wi_changed[:, [0, 1, 2, 3, 4, 5]] = 0
    # No city
    elif variable_index ==8:
        wi_changed[:, variable_index-2] =0

    # Calculate the new alpha for all motivations with the changed customer characteristic
    alpha_changed = []
    sum_alpha_changed = 0
    for i in range(len(gamma_data)):
        alpha_i = np.exp(kappa_mean[i] + np.dot(xib_data.mean(axis=0), beta_data.iloc[i]) + np.dot(wi_changed.mean(axis=0), gamma_data.iloc[i]))
        alpha_changed.append(alpha_i)
        sum_alpha_changed += alpha_i

    # Calculate the shifted theta for motivation of interest 
    theta_shifted = alpha_changed[m] / sum_alpha_changed

    # Calculate the baseline theta for motivation of interest 
    theta_baseline = alpha_baseline[m] / sum_alpha_baseline

    # Calculate the odds ratio
    odds_ratio = theta_shifted / theta_baseline

    return odds_ratio

# Calculate odds ratios for all motivations and variables
odds_ratios = np.zeros((len(gamma_data), len(hi_data.columns)+2))

# For every motivation, and for each variable 
for m in range(len(gamma_data)):
    for variable_index in range(len(hi_data.columns)+2):
        odds_ratio = calculate_odds_ratio(kappa_mean, beta_data, gamma_data, xib_data, hi_data, m, variable_index)
        odds_ratios[m, variable_index] = odds_ratio

# Print odds ratios for all motivations and variables
for m in range(len(gamma_data)):
    for variable_index in range(len(hi_data.columns)+2):
        print(f"The odds ratio for shopping motivation {m} with dummy variable {variable_index} set to 1 in customer characteristic (w) is: {odds_ratios[m, variable_index]}")

# Calculate summary statistics
average_odds_ratio_per_variable = np.mean(odds_ratios, axis=0)
min_odds_ratio_per_variable = np.min(odds_ratios, axis=0)
max_odds_ratio_per_variable = np.max(odds_ratios, axis=0)

# Print summary statistics
for variable_index in range(len(hi_data.columns)+2):
    print(f"Variable {variable_index}:")
    print(f"  Average odds ratio: {average_odds_ratio_per_variable[variable_index]}")
    print(f"  Minimum odds ratio: {min_odds_ratio_per_variable[variable_index]} (Motivation {np.argmin(odds_ratios[:, variable_index])})")
    print(f"  Maximum odds ratio: {max_odds_ratio_per_variable[variable_index]} (Motivation {np.argmax(odds_ratios[:, variable_index])})")
