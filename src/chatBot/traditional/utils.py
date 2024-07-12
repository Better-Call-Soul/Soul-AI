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
    # print(text_to_vector_v1)
    # print(text_to_vector_v2)

    # distance of similarity
    cosine = distance.cosine(text_to_vector_v1, text_to_vector_v2)
    return round((1-cosine),2)


if __name__ ==  "__main__":
    '''
    this is a testing unit of fitting the dataset or fitting on the fly
    '''
    cos1 = cosine_distance_countvectorizer_method('hany am ahmed hany', "hany am mama dada hany")
    cos2 = cosine_distance_countvectorizer_method('hany am ahmed hany', "ahmed is an hany question, can you hany that you am?")
    print(cos1, cos2)

    vectorizer = CountVectorizer()
    sen = vectorizer.fit_transform(["hany am mama dada hany", "ahmed is an hany question, can you hany that you am?"])
    text_to_vector_v1 = sen.toarray()[0].tolist()
    text_to_vector_v2 = sen.toarray()[1].tolist()

    sen1 = vectorizer.transform(['hany am ahmed hany', "hany am mama dada hany"])
    sen2 = vectorizer.transform(['hany am ahmed hany', "ahmed is an hany question, can you hany that you am?"])
    text_to_vector_v3 = sen1.toarray()[0].tolist()
    text_to_vector_v4 = sen1.toarray()[1].tolist()
    text_to_vector_v5 = sen2.toarray()[0].tolist()
    text_to_vector_v6 = sen2.toarray()[1].tolist()
    print(text_to_vector_v3, text_to_vector_v4)
    print(text_to_vector_v5, text_to_vector_v6)

    cosine1 = distance.cosine(text_to_vector_v3, text_to_vector_v4)
    cosine2 = distance.cosine(text_to_vector_v5, text_to_vector_v6)
    print(round((1-cosine1),2), round((1-cosine2),2))

