import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance

def map_tag_pattern(df, tag_col, text_col, res_col):
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