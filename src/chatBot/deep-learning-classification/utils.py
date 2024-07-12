import random
from constants import preprocessing_steps

def map_tag_pattern(preprocessor, df, text_col, res_col):
  train_data = []
  train_labels = []

  for index, item in df.iterrows():
      ptrns = item[text_col]
      rspns = item[res_col]
      for j in range(len(ptrns)):
          cleaned_line = preprocessor.clean(ptrns[j], preprocessing_steps, '')[0]
          train_data.append(cleaned_line)
          cleaned_label = preprocessor.clean(random.choice(rspns), preprocessing_steps, '')[0]
          train_labels.append(cleaned_label)

  return train_data, train_labels


