print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!INTRODUCTION TO PANDAS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## TASK 1 ##
df = pd.read_csv('dataset/housing.csv')

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Handle missing values(by dropping the missing values)
df = df.dropna()

print("\nSummary of the dataset:")
print(df.describe())

# Indexing of the file
# Label-based indexing
subset_label_based = df[['crim', 'rm', 'age', 'tax']]  
# Position-based indexing
subset_position_based = df.iloc[:, [0, 4, 5, 8]]  

# new DataFrame by filtering rows based on a condition
condition = df['age'] > 60  
filtered_df = df[condition]

# Display the new DataFrames
print("\nSubset of columns using label-based indexing:")
print(subset_label_based.head())

print("\nSubset of columns using position-based indexing:")
print(subset_position_based.head())

print("\nNew DataFrame after filtering rows based on a condition:")
print(filtered_df.head())


## TASK 2 ##
# Identify missing values
missing_values = df.isnull().sum()

# Input missing numerical values with the mean 
mean_imputation_cols = ['age', 'tax']  
categorical_remove_cols = ['b']  

# Input missing numerical values with mean
df[mean_imputation_cols] = df[mean_imputation_cols].fillna(df[mean_imputation_cols].mean())

# Remove rows with missing categorical values
df = df.dropna(subset=categorical_remove_cols)

# Creating a new column by applying a mathematical operation
df['new_column'] = df['crim'] * df['zn']  

# Convert a categorical variable into numerical representation using one-hot encoding
df = pd.get_dummies(df, columns=['rad'], prefix='one_hot')  

# Group the data 
grouped_data = df.groupby('ptratio')

# Apply aggregation functions (sum, mean, count) to the grouped data
aggregated_data = grouped_data.agg({'C1': ['sum', 'mean'], 'C2': 'count'})

# Present the results
print("\nAggregated Data:")
print(aggregated_data)

## TASK 3 ##
housing = pd.read_csv('dataset/housing.csv')
salary = pd.read_csv('dataset/salary_dataset.csv')

# Display the original datasets
print("Housing Dataset:")
print(housing.head())

print("\nSalary Dataset:")
print(salary.head())

# Merge using different types of joins
# Inner Join
inner_merge = pd.merge(housing, salary, on='age', how='inner')

# Outer Join
outer_merge = pd.merge(housing, salary, on='age', how='outer')

# Left Join
left_merge = pd.merge(housing, salary, on='age', how='left')

# Right Join
right_merge = pd.merge(housing, salary, on='age', how='right')

# Display the merged datasets
print("\nInner Merge:")
print(inner_merge.head())

print("\nOuter Merge:")
print(outer_merge.head())

print("\nLeft Merge:")
print(left_merge.head())

print("\nRight Merge:")
print(right_merge.head())

## TASK 4 ##
# Bar plot
housing['age'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Line plot 
housing['crim'].plot(kind='line', color='green', marker='o', linestyle='-', linewidth=2)
plt.title('Crime Rate over Time')
plt.xlabel('Index')
plt.ylabel('Crime Rate')
plt.show()

# Scatter plot 
housing.plot.scatter(x='rm', y='medv', c='blue', alpha=0.5)
plt.title('Scatter Plot of Room Number vs. Median Value')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Median Value of Homes')
plt.show()

# Visualize correlation matrix
corr_matrix = housing.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Histograms 
housing.hist(bins=20, figsize=(12, 10), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numerical Columns', y=1.02)
plt.show()

# Box plots 
housing.plot(kind='box', vert=False, figsize=(12, 8), color='skyblue')
plt.title('Box Plots of Numerical Columns')
plt.show()

print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!EXPLORING NUMPY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
## TASK 5 ##
import numpy as np
arr = np.arange(1, 11)
arr2 = np.arange(11, 21)

# Add
addition_result = arr + arr2
# Subtract
subtraction_result = arr - arr2
# Multiply
multiplication_result = arr * arr2
# Divide
division_result = arr / arr2

# Print the results
print("Array 'arr':", arr)
print("Array 'arr2':", arr2)

print("\nAddition Result:")
print(addition_result)

print("\nSubtraction Result:")
print(subtraction_result)

print("\nMultiplication Result:")
print(multiplication_result)

print("\nDivision Result:")
print(division_result)

## TASK 6 ##
# Reshape 'arr' into a 2x5 matrix.
arr_reshaped = arr.reshape(2, 5)

# Transpose the matrix 
arr_transposed = arr_reshaped.T

# Flatten the transposed matrix into a 1D array.
arr_flattened = arr_transposed.flatten()

# Stack 'arr' and 'arr2' vertically
stacked_vertical = np.vstack((arr, arr2))

# Print the results
print("Original 'arr':")
print(arr)

print("\nReshaped 'arr' (2x5 matrix):")
print(arr_reshaped)

print("\nTransposed Matrix:")
print(arr_transposed)

print("\nFlattened Transposed Matrix:")
print(arr_flattened)

print("\nStacked 'arr' and 'arr2' vertically:")
print(stacked_vertical)

## TASK 7 ##
# mean
arr_mean = np.mean(arr)
# median
arr_median = np.median(arr)
# standard deviation
arr_std = np.std(arr)

# Find the maximum and minimum values
arr_max = np.max(arr)
arr_min = np.min(arr)

# Normalize 'arr'
arr_normalized = (arr - arr_mean) / arr_std

# Print the results
print("Original 'arr':")
print(arr)

print("\nMean of 'arr':", arr_mean)
print("Median of 'arr':", arr_median)
print("Standard Deviation of 'arr':", arr_std)

print("\nMaximum value in 'arr':", arr_max)
print("Minimum value in 'arr':", arr_min)

print("\nNormalized 'arr':")
print(arr_normalized)

## TASK 8 ##
# Boolean array 'bool_arr' for elements in 'arr' greater than 5.
bool_arr = arr > 5

# Extract the elements from 'arr' that are greater than 5.
filtered_elements = arr[bool_arr]

# Print the results
print("Original 'arr':")
print(arr)

print("\nBoolean Array 'bool_arr' for elements greater than 5:")
print(bool_arr)

print("\nElements in 'arr' greater than 5:")
print(filtered_elements)

## TASK 9 ##
# Generating a 3x3 matrix with random values between 0 and 1.
random_matrix = np.random.rand(3, 3)

# An array of 10 random integers between 1 and 100.
random_integers = np.random.randint(1, 101, 10)

# Shuffle the elements of 'arr' randomly.
shuffled_arr = np.random.permutation(arr)

# Print the results
print("Random Matrix (3x3) between 0 and 1:")
print(random_matrix)

print("\nArray of 10 Random Integers between 1 and 100:")
print(random_integers)

print("\nShuffled 'arr':")
print(shuffled_arr)

## TASK 10 ##
# Square root function
arr_sqrt = np.sqrt(arr)

# Exponential function
arr_exp = np.exp(arr)

# Print the results
print("Original 'arr':")
print(arr)

print("\nSquare Root of 'arr':")
print(arr_sqrt)

print("\nExponential Function (e^x) applied to 'arr':")
print(arr_exp)

## TASK 11 ##
# 3x3 matrix
mat_a = np.random.rand(3, 3)

# 3x1 matrix
vec_b = np.random.rand(3, 1)

# Multiply using the dot product.
result = np.dot(mat_a, vec_b)

# Print the results
print("Matrix 'mat_a' (3x3):")
print(mat_a)

print("\nMatrix 'vec_b' (3x1):")
print(vec_b)

print("\nResult of mat_a * vec_b (Dot Product):")
print(result)

## TASK 12 ##
# 2D array 'matrix' with values from 1 to 9.
matrix = np.arange(1, 10).reshape(3, 3)

# Mean of each row
row_means = matrix.mean(axis=1, keepdims=True)

# Subtract the mean of each row from each element in that row using broadcasting
result = matrix - row_means

# Print the results
print("Original 'matrix' (2D array):")
print(matrix)

print("\nRow Means:")
print(row_means)

print("\nResult after Subtracting Row Means (Broadcasting):")
print(result)