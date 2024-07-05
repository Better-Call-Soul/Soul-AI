import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance

def map_tag_pattern(df, tag_col, text_col, res_col):
  '''
  Map the dataset to the required format for training
  :param df: pd.DataFrame: The dataset
  :type df: pd.DataFrame
  :param tag_col: str: The column name for the tag
  :type tag_col: str
  :param text_col: str: The column name for the text
  :type text_col: str
  :param res_col: str: The column name for the response
  :type res_col: str
  :return: The mapped dataset
  :rtype: pd.DataFrame
  '''
  dic = {tag_col:[], text_col:[], res_col:[]}

  for index, item in df.iterrows():
      ptrns = item[text_col]
      rspns = item[res_col]
      tag = item[tag_col]
      for j in range(len(ptrns)):
          dic[tag_col].append(tag)
          dic[text_col].append(ptrns[j])
          dic[res_col].append(rspns)

  return pd.DataFrame.from_dict(dic)

def cosine_distance_countvectorizer_method(s1, s2):
    '''
    Calculate the cosine distance between two sentences
    :param s1: str: The first sentence
    :type s1: str
    :param s2: str: The second sentence
    :type s2: str
    :return: The cosine similarity between the two sentences
    :rtype: float
    '''
    # sentences to list
    allsentences = [s1 , s2]

    # text to vector
    vectorizer = CountVectorizer()
    all_sentences_to_vector = vectorizer.fit_transform(allsentences)
    text_to_vector_v1 = all_sentences_to_vector.toarray()[0].tolist()
    text_to_vector_v2 = all_sentences_to_vector.toarray()[1].tolist()

    # distance of similarity
    cosine = distance.cosine(text_to_vector_v1, text_to_vector_v2)
    return round((1-cosine),2)