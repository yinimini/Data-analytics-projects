import pandas as pd
import matplotlib.pyplot as plt
from typing import List

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the Excel file into a Pandas DataFrame.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: DataFrame containing the Excel data.
    """
    return pd.read_excel(file_path)


def add_days_in_cycle_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Days_in_cycles' column to the DataFrame that represents 
    the day number within each cycle.

    Args:
        df (pd.DataFrame): DataFrame with a 'Cycle' column.

    Returns:
        pd.DataFrame: DataFrame with the new 'Days_in_cycles' column added.
    """
    cycles = df['Cycle'].tolist()
    day = 1
    cycle_days = []  
    prev_cycle = cycles[0]

    for cycle in cycles:
        if prev_cycle == cycle:
            cycle_days.append(day)
        else:
            day = 1
            prev_cycle = cycle
            cycle_days.append(day)
        day += 1

    df['Days_in_cycles'] = pd.DataFrame(cycle_days)
    return df


def plot_symptom_recurrence(df: pd.DataFrame, symptoms: List[str], total_cycles: int) -> None:
    """
    Plot the recurrence of symptoms over days in the cycle.

    Args:
        df (pd.DataFrame): DataFrame containing symptom data.
        symptoms (List[str]): List of symptom column names.
        total_cycles (int): Total number of cycles to be displayed in the title.

    Returns:
        None
    """
    df.plot(x='Days_in_cycles', y=symptoms, kind='bar')
    plt.xlabel(f'Day in each cycle, total cycles = {total_cycles}')
    plt.ylabel('1=YES, 0=NO')
    plt.title('Recurrence of symptoms')
    plt.legend()
    plt.show()


def calculate_symptom_probability(symptom_list: pd.Series) -> float:
    """
    Calculate the probability of a symptom occurring in the dataset.

    Args:
        symptom_list (pd.Series): Series containing binary values representing symptom presence (1) or absence (0).

    Returns:
        float: Probability of the symptom occurrence.
    """
    positive_symp = [i for i in symptom_list if i == 1]
    probability = len(positive_symp) / len(symptom_list)
    return probability


def calculate_and_plot_probabilities(df: pd.DataFrame, symptom_columns: List[str]) -> None:
    """
    Calculate and plot the probabilities of symptom occurrence.

    Args:
        df (pd.DataFrame): DataFrame containing symptom data.
        symptom_columns (List[str]): List of symptom column names.

    Returns:
        None
    """
    symp_probabilities = []
    for symptom in symptom_columns:
        probability = calculate_symptom_probability(df[symptom])
        symp_probabilities.append(probability)
        print(f'Probability of getting {symptom} during {len(df[symptom])} days/7 cycles is {round(probability * 100, 3)} %')

    # Plotting probabilities
    plt.bar(symptom_columns, symp_probabilities)
    plt.xlabel('Symptoms')
    plt.ylabel('Probability')
    plt.title('Probabilities of symptom occurrence')
    plt.show()


def save_dataframe_to_excel(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the DataFrame to an Excel file.

    Args:
        df (pd.DataFrame): DataFrame to be saved.
        file_path (str): Path where the Excel file will be saved.

    Returns:
        None
    """
    df.to_excel(file_path)


if __name__ == "__main__":
    # Load the data
    file_path = "Menstration_symp.xlsx"
    df = load_data(file_path)

    # Add 'Days_in_cycles' column
    df = add_days_in_cycle_column(df)

    # Define the symptoms to be analyzed
    symptoms = ['Cramps', 'Tender_breasts', 'Low_energy', 'Headache', 'Abdominal_pain', 'Nausea']

    # Plot the symptom recurrence
    plot_symptom_recurrence(df, symptoms, total_cycles=7)

    # Calculate and plot probabilities of symptom occurrence
    calculate_and_plot_probabilities(df, symptoms)

    # Save the updated DataFrame to a new Excel file
    save_path = "Menstration_symp.xlsx"
    save_dataframe_to_excel(df, save_path)





