
print("hello")

import pandas as pd
print("hello1")
import matplotlib.pyplot as plt
print("hello1")
import seaborn as sns
print("hello1")
def load_data():
    """
    Load the breast cancer dataset from a CSV file into a DataFrame.
    """
    df = pd.read_csv('breast-cancer.csv')
    return df

def summarize_data(df):
    """
    Provide a basic summary of the dataset.
    """
    summary = {
        'Number of Rows': df.shape[0],
        'Number of Columns': df.shape[1],
        'Missing Values': df.isnull().sum().to_dict(),
        'Data Types': df.dtypes.to_dict()
    }
    return summary

def calculate_statistics(df):
    """
    Calculate mean, median, and mode for each numeric column in the DataFrame.
    """
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        print(column, 'Mean', df[column].mean(), 'Median', df[column].median(), 'Mode', df[column].mode()[0])

df = load_data()

calculate_statistics(df)

def visualize_data(df):
    """
    Visualize the dataset using individual histograms and box plots for each numeric column.
    """
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Create histograms
    for column in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[column], bins=30, kde=True)
        plt.title(f'Histogram of {column}')
        plt.show()
    
    # Create box plots
    for column in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[column])
        plt.title(f'Box Plot of {column}')
        plt.show()

# visualize_data(df)

def correlation_analysis(df):
    """
    Perform correlation analysis and visualize the correlation matrix.
    """
    numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Filter numeric columns
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

def pairplot(df):
    sns.pairplot(df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'diagnosis']], hue='diagnosis')
    plt.show()

def boxPlot(df):
    x_values = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                'smoothness_mean', 'compactness_mean', 'concavity_mean',
                'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                'fractal_dimension_se', 'radius_worst', 'texture_worst',
                'perimeter_worst', 'area_worst', 'smoothness_worst',
                'compactness_worst', 'concavity_worst', 'concave points_worst',
                'symmetry_worst', 'fractal_dimension_worst']

    y_value = 'radius_mean'

    n_rows = 10
    n_cols = 3

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 60))

    for i, x_value in enumerate(x_values):
        ax = axes.flatten()[i]
        sns.boxplot(data=df, x='diagnosis', y=x_value, hue='diagnosis', ax=ax, palette=["#006992", "#ff7d00"])
        ax.set_title(f'{x_value.capitalize()} by Diagnosis')
        ax.set_ylabel(x_value.capitalize())

    plt.tight_layout()
    plt.show()

# Call the boxPlot function to generate and display the box plots
boxPlot(df)

