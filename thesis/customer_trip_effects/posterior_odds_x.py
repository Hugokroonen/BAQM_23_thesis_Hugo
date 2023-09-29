"""
Description:
    In this script, we calculate the posterior odds ratios for the trip-specific variables. We also analyse the resulting
    trip-specific effects. 
"""

import numpy as np
import pandas as pd

# Posterior odds for a variable in X

# Load gamma data
gamma_data = pd.read_csv("output/M30/CTM/gamma.csv")

# Load beta data
beta_data = pd.read_csv("output/M30/CTM/beta.csv")

# Load kappa data
kappa_data = pd.read_csv("output/M30/CTM/kappa.csv")

# Load xib_data
xib_data = pd.read_csv("data/x.csv")

# Load hi_data 
hi_data = pd.read_csv("data/h.csv")

# Population-level mean of motivation intercepts
kappa_mean = np.mean(kappa_data, axis=0)

def calculate_odds_ratio(kappa_mean, beta_data, gamma_data, x_data, hi_data, m, variable_index):

    # Calculate the baseline alpha for all motivations
    alpha_baseline = []
    sum_alpha_baseline = 0
    for i in range(len(gamma_data)):
        alpha_i = np.exp(kappa_mean[i] + np.dot(x_data.mean(axis=0), beta_data.iloc[i]) + np.dot(hi_data.mean(axis=0), gamma_data.iloc[i]))
        alpha_baseline.append(alpha_i)
        sum_alpha_baseline += alpha_i
    
    x_changed = np.copy(x_data)

    # Set one trip-specific characteristic (x) to a specific level 

    # Basket discount percentage
    if variable_index == 0:  # For the first column of X
        x_changed[:, variable_index] *= 1.5  # Apply 50% increase shock

    # Set all other variables to 1 
    elif variable_index in [1, 2]:
        x_changed[:, variable_index] = 1
    # Not weekend 
    elif variable_index == 3:
        x_changed[:, variable_index-2] = 0
    # Before 5PM
    elif variable_index == 4:
        x_changed[:, variable_index-2] = 0

    # Calculate the new alpha for all motivations with the changed customer characteristic
    alpha_changed = []
    sum_alpha_changed = 0
    for i in range(len(gamma_data)):
        alpha_i = np.exp(kappa_mean[i] + np.dot(x_changed.mean(axis=0), beta_data.iloc[i]) + np.dot(hi_data.mean(axis=0), gamma_data.iloc[i]))
        alpha_changed.append(alpha_i)
        sum_alpha_changed += alpha_i

    # Calculate the shifted theta for motivation m
    theta_shifted = alpha_changed[m] / sum_alpha_changed

    # Calculate the baseline theta for motivation m
    theta_baseline = alpha_baseline[m] / sum_alpha_baseline

    # Calculate the odds ratio
    odds_ratio = theta_shifted / theta_baseline

    return odds_ratio

# Calculate odds ratios for all motivations and variables in X
odds_ratios_x = np.zeros((len(gamma_data), len(xib_data.columns)+2))

for m in range(len(gamma_data)):
    for variable_index in range(len(xib_data.columns)+2):
        odds_ratio_x = calculate_odds_ratio(kappa_mean, beta_data, gamma_data, xib_data, hi_data, m, variable_index)
        odds_ratios_x[m, variable_index] = odds_ratio_x

# Print odds ratios for all motivations and variables in X
for m in range(len(gamma_data)):
    for variable_index in range(len(xib_data.columns)+2):
        print(f"The odds ratio for shopping motivation {m} with dummy variable {variable_index} set to 1 in customer characteristic (x) is: {odds_ratios_x[m, variable_index]}")

# Calculate summary statistics for X
average_odds_ratio_per_variable_x = np.mean(odds_ratios_x, axis=0)
min_odds_ratio_per_variable_x = np.min(odds_ratios_x, axis=0)
max_odds_ratio_per_variable_x = np.max(odds_ratios_x, axis=0)

# Print summary statistics for X
for variable_index in range(len(xib_data.columns)+2):
    print(f"Variable {variable_index} in X:")
    print(f"  Average odds ratio: {average_odds_ratio_per_variable_x[variable_index]}")
    print(f"  Minimum odds ratio: {min_odds_ratio_per_variable_x[variable_index]} (Motivation {np.argmin(odds_ratios_x[:, variable_index])})")
    print(f"  Maximum odds ratio: {max_odds_ratio_per_variable_x[variable_index]} (Motivation {np.argmax(odds_ratios_x[:, variable_index])})")
