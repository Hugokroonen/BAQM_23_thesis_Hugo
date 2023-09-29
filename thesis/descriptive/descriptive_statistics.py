import pandas as pd

# Read the CSV file into a pandas DataFrame
x = pd.read_csv('data/x.csv')

# Calculate the average of the first column
average_first_column = x.iloc[:, 0].mean()
average_second_column = x.iloc[:, 1].mean()
average_third_column = x.iloc[:, 2].mean()


print(f"Average discount: {average_first_column*100}")
print(f"Average Weekend: {average_second_column*100}")
print(f"Average After 5PM: {average_third_column*100}")

# Read the CSV file into a pandas DataFrame
h = pd.read_csv('data/h.csv')

# Calculate the average of the first column
average_city_column = h.iloc[:, 6].mean()
unknown_age = h.iloc[:, 0].mean()
age_0_25 = h.iloc[:, 1].mean()
age_25_35 = h.iloc[:, 2].mean()
age_35_45 = h.iloc[:, 3].mean()
age_55_65 = h.iloc[:, 4].mean()
age_65 = h.iloc[:, 5].mean()

print(f"Average city: {average_city_column*100}")
print(f"Average unknown: {unknown_age*100}")
print(f"Average 0-25: {age_0_25*100}")
print(f"Average 25-35: {age_25_35*100}")
print(f"Average 35-45: {age_35_45*100}")
print(f"Average 55-65: {age_55_65*100}")
print(f"Average 65+: {age_65*100}")

h = pd.read_csv('data/h.csv', header=None)  # No header information provided
y = pd.read_csv('data/y.csv', header=None)
final_sample = pd.read_csv('data/products.csv')

# Find the top 10 most occurring values in the third column of y
top_product_ids = y.iloc[:, 2].value_counts().head(10).index.tolist()

# Filter final_sample DataFrame to match top product_ids and retrieve lowest_levels
filtered_final_sample = final_sample[final_sample['product_id'].isin(top_product_ids)]
product_id_lowest_level = filtered_final_sample[['product_id', 'lowest_level']].drop_duplicates()

# Count unique baskets for each top product
basket_counts = {}
for product_id in top_product_ids:
    baskets_ith_product = y[y.iloc[:, 2] == product_id].iloc[:, 1].nunique()  # Assuming second column contains basket IDs
    basket_counts[product_id] = baskets_with_product

# Calculate percentages
total_baskets = y.iloc[:, 1].nunique()  # Assuming second column contains basket IDs
percentages = {product_id: (count / total_baskets) * 100 for product_id, count in basket_counts.items()}

print("Top 10 most occurring product IDs with their lowest levels and basket percentages:")
for product_id, lowest_level in product_id_lowest_level.values:
    print(f"Product ID: {product_id}, Lowest Level: {lowest_level}, Basket Percentage: {percentages.get(product_id, 0):.2f}%")







