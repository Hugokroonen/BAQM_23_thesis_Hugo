"""
Description:
    In this script, we calculate the distinctiveness of each motivation based on the cosine distance, as well as the overall
    distinctiveness of all motivations to be used as a model selection criterion. 
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# Load the counts_phi data
data = pd.read_csv('output/M30/CTM/counts_phi.csv', header=None)

# Get the number of motivations
num_motivations = data.shape[1]

# Initialize an empty dictionary to store the minimum cosine distances
min_distances = {}

# Iterate over each motivation
for motivation in range(num_motivations):
    motivation_probabilities = data.iloc[:, motivation].values
    
    # Compare the current motivation to other motivations
    min_distance = float('inf')
    most_similar_motivation = None
    
    for other_motivation in range(num_motivations):
        if other_motivation == motivation:
            continue  # Skip comparing the motivation to itself
        
        other_probabilities = data.iloc[:, other_motivation].values
        
        # Calculate the cosine distance between the probabilities
        distance = cosine_distances([motivation_probabilities], [other_probabilities])[0][0]
        
        # Check if the current distance is smaller than the previous minimum
        if distance < min_distance:
            min_distance = distance
            most_similar_motivation = other_motivation
    
    # Store the minimum distance and the most similar motivation in the dictionary
    min_distances[motivation] = {'min_distance': min_distance, 'most_similar_motivation': most_similar_motivation}

# Print the minimum cosine distance and the most similar motivation for each motivation
for motivation, info in min_distances.items():
    min_distance = info['min_distance']
    most_similar_motivation = info['most_similar_motivation']
    print(f"Minimum cosine distance for motivation {motivation}: {min_distance}")
    print(f"Most similar motivation: {most_similar_motivation}")
    print()

# Calculate the average of all minimum cosine distances
average_min_distance = np.mean([info['min_distance'] for info in min_distances.values()])

# Print the average minimum cosine distance
print(f"Average minimum cosine distance: {average_min_distance}")
