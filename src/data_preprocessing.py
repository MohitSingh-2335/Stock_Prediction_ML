# src/data_preprocessing.py

import pandas as pd

def load_and_clean_data(file_path):
    """
    Loads the dataset from a CSV file, converts the timestamp,
    and sorts the data.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The cleaned and sorted DataFrame.
    """
    # Load the dataset from the provided file path
    df = pd.read_csv(file_path)

    # Convert the 'timestamp' column to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y:%m:%d %H:%M:%S')

    # Sort the DataFrame by the timestamp to ensure chronological order
    df = df.sort_values('timestamp')

    return df