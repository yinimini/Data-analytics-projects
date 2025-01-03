import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import missingno as msno
import seaborn as sns

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the Excel file into a Pandas DataFrame.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: DataFrame containing the Excel data.
    """
    raw_data = pd.read_csv(file_path)
    print(raw_data.info)
    return raw_data

critical_features = [ 
    'Cramps', 
    'Tender_breasts', 
    'Low_energy', 
    'Backache', 
    'Headache', 
    'Acne', 
    'Insomnia', 
    'Vaginal_itching', 
    'Vaginal_dryness', 
    'Nausea', 'Bloating', 
    'Diarrhea', 
]

class DataExplorer:
    def __init__(self, data: pd.DataFrame):
        """Initialize with a pandas DataFrame."""
        self.data = data

    def missing_values_summary(self) -> pd.Series:
        """Get summary of missing values in the raw dataset.

        Returns:
            pd.Series: Summary of missing values.
        """
        missing_summary = self.data.isnull().sum()
        print(f'The information about missing values in the raw data is:\n{missing_summary}')
        return missing_summary

    def plot_missing_values_matrix(self, output_path_plot:str):
        """Visualize missing values.
        Args:
            output_path_path (str): Output path for where the plot is going to be. 
        """
        msno.matrix(self.data, sparkline=False, figsize=(8, 6), fontsize=11, color=(0.7, 0.57, 0.47))
        gray_patch = mpatches.Patch(color='#B29177', label='Data present')
        white_patch = mpatches.Patch(color='white', label='Data absent')
        plt.legend(handles=[gray_patch, white_patch], fontsize=10)
        plt.title('Missing Values Matrix')
        plt.savefig(output_path_plot)
        plt.show()
        

    def data_summary(self) -> pd.DataFrame:
        """Get summary of raw data

        Returns:
            pd.DataFrame: Summary of raw data contaning info like count, mean,
            std, min and etc.
        """
        summary = self.data.describe()
        print(f'Summary of raw data:\n{summary}')
        return summary

    def column_names(self) -> list[str]:
        """Retrieve all the columns in the raw dataset

        Returns:
            list[str]: All the columns in the dataset
        """
        names = self.data.columns.tolist()
        print(f'Raw data has these columns:\n{names}')
        return names

    def plot_heatmap(self, critical_features:list[str], output_path_plot:str):
        """Visualization of symptions correlations

        Args:
            critical_features (list[str]): Critical features for menstrual cycle.
            output_path_path (str): Output path for where the plot is going to be. 
        """
        data_with_critical_features = self.data[critical_features]
        plt.figure(figsize=(12, 8))
        sns.heatmap(data_with_critical_features.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Heatmap of Symptoms Correlations')
        plt.savefig(output_path_plot)
        plt.show()
        



if __name__ == "__main__":
    file_path = "Menstration_symp_1.csv"
    raw_data = load_data(file_path)
    explorer = DataExplorer(raw_data)
    explorer.missing_values_summary()
    explorer.plot_missing_values_matrix('/Users/Yini Chen/Documents/Data-analytics-projects/Menstrual_cycle_analysis/Plots/missing_values_plot.jpeg')
    explorer.data_summary()
    explorer.column_names()
    explorer.plot_heatmap(critical_features, '/Users/Yini Chen/Documents/Data-analytics-projects/Menstrual_cycle_analysis/Plots/symptoms_correlation_heatmap.jpeg')
    







