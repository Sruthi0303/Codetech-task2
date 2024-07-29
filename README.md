# Codetech-task2
Name: K SRUTHI

Company: CODETECH IT SOLUTIONS

ID:CT6DS710

Domain: DATA SCIENCE

Duration:June 25th to August 10th,2024

Mentor:Muzammil Ahmed

# Project:Preditive Modeling with Linear Regression on Car Weight vs MPG Analysis

# Overview of the project

This project involves analyzing the relationship between car weight and miles per gallon (MPG) using a dataset. The analysis includes data loading, exploratory data visualization, and linear regression modeling to understand and quantify the correlation between these two variables.

## Objective

The goal of this project is to explore how car weight impacts MPG and to visualize this relationship through linear regression analysis.

## Key Activities

1. **Data Loading and Exploration**
   - **Load Dataset:** The car dataset is loaded from the file `auto-mpg.csv` into a Pandas DataFrame.
   - **Initial Inspection:**
     - `auto.head()`: Displays the first few rows of the dataset.
     - `auto.info()`: Provides summary information about the dataset, including column data types and non-null counts.
     - `auto.describe()`: Offers descriptive statistics for numeric columns.

2. **Data Visualization**
   - **Scatter Plot of Weight vs. MPG:**
     - A scatter plot is created to visualize the relationship between car weight and MPG, with weight on the x-axis and MPG on the y-axis.
   - **Scatter Plot with Hardcoded Linear Regression Line:**
     - Adds a hardcoded red line to the scatter plot to illustrate a linear trend. Note that this line is for illustrative purposes and may not accurately reflect the regression line.

3. **Linear Regression Modeling**
   - **Prepare Data for Modeling:**
     - Defines car weight as the feature (`X`) and MPG as the target variable (`Y`).
   - **Build and Train Model:**
     - Uses `LinearRegression` from Scikit-Learn to create and fit a linear regression model.
   - **Visualize Regression Results:**
     - Plots the regression line based on model predictions overlaid on the scatter plot to visualize the fit of the linear regression model.

## Technologies Used

- **Pandas:** For data manipulation and exploration.
- **Matplotlib:** For creating visualizations, including scatter plots.
- **NumPy:** For numerical operations (implicitly used).
- **Scikit-Learn (sklearn):** For building and applying the linear regression model.

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
auto = pd.read_csv('auto-mpg.csv')

# Display initial data and statistics
print(auto.head())
print(auto.info())
print(auto.describe())

# Scatter plot of Weight vs. MPG
plt.figure(figsize=(10,10))
plt.scatter(auto['weight'], auto['mpg'])
plt.title('Miles per Gallon vs. Weight of Car')
plt.xlabel('Weight of Car')
plt.ylabel('Miles per Gallon')
plt.show()

# Scatter plot with hardcoded regression line
plt.figure(figsize=(10,10))
plt.scatter(auto['weight'], auto['mpg'])
plt.plot(auto['weight'], (auto['weight'] / -105) + 55, c='red')  # Hardcoded line
plt.title('Miles per Gallon vs. Weight of Car')
plt.xlabel('Weight of Car')
plt.ylabel('Miles per Gallon')
plt.show()

# Linear Regression Modeling
X = auto[['weight']]
Y = auto['mpg']
MPG_Pred = LinearRegression()
MPG_Pred.fit(X, Y)

# Scatter plot with regression line
plt.figure(figsize=(10,10))
plt.scatter(auto['weight'], auto['mpg'])
plt.scatter(X, MPG_Pred.predict(X), c='Red')  # Predicted values
plt.title('Miles per Gallon vs. Weight of Car')
plt.xlabel('Weight of Car')
plt.ylabel('Miles per Gallon')
plt.show()

# Alternative scatter plot with regression line
plt.figure(figsize=(10,10))
plt.scatter(auto['weight'], auto['mpg'])
plt.scatter(X, MPG_Pred.predict(X), c='Red')  # Predicted values
plt.title('Miles per Gallon vs. Weight of Car')
plt.xlabel('Weight of Car')
plt.ylabel('Miles per Gallon')
plt.show()
