import pandas as pd
from datasets import Dataset

def preprocess_data(file_name, col_name, max_sequence_length=11000, num_samples=None):
  """
  Preprocess the data to the desired format
  Args:
    questions_col (str): The name of the column containing the questions
    answers_col (str): The name of the column containing the answers
    max_sequence_length (int): The maximum length of the sequence
    num_samples (int): The number of samples to select from the dataset
  Returns:
    dataset: The preprocessed dataset contains the text column
  """
  df = pd.read_csv(file_name)

  # Calculate the length of each row
  df['row_length'] = df['text'].apply(len)

  # Filter out rows with length more than max_sequence_length (i.e. 11000) to obtain just acceptable length of input
  df_filtered = df[df['row_length'] <= max_sequence_length]

  # drop un needed columns
  df_filtered = df_filtered.drop(['row_length'], axis=1)

  # convert to dataset and select num_samples (i.e. 1000) record for now (considering memory resource)
  dataset = Dataset.from_pandas(df_filtered)

  if num_samples and len(dataset) >= num_samples:
    dataset = dataset.select(range(num_samples))

  return dataset