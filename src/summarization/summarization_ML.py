import re
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from preprocessing.preprocessing import Preprocessing

# Class for machine learning based summarization of text using tf-idf algorithm
class MachineLearningSummarization:
    def __init__(self,threshold=0.9):
        self.threshold=threshold
        self.preprocess=Preprocessing()
    # Clean the text
    def clean_text(self,text):
        text=self.preprocess.coreference_resolution(text)
        statements = re.findall(r'[^.!?]+[.!?]', text)
        sentences =[]
        # remove non-english characters and extra whitespaces
        for sentence in statements:
            sentence=self.preprocess.preprocess_text([sentence])[0]
            sentences.append(sentence)

        return sentences
    
    # Count the number of words in the text
    def count_words(self,text):
        # Tokenize the text
        words = word_tokenize(text)
        words=[word for word in words if word not in  stopwords.words('english')] # remove non-alphabetic characters
        return len(words)
    
    # Count the number of words in each sentence
    def count_in_sentences(self, sentences):
        count=[]
        for i,sentence in enumerate(sentences):
            temp={'id':i,'word_count':self.count_words(sentence[:-1])}
            count.append(temp)
        return count
    
    # Create a frequency dictionary for each word of the sentence
    def freq_dict(self,sentences):
        freq_list=[]
        for i,sentence in enumerate(sentences):
            words = word_tokenize(sentence[:-1]) # tokenize the sentence
            freq_dict = {}
            for word in words:
                if(word in stopwords.words('english')): # remove stop words
                    continue
                word=word.lower() # convert to lower case
                if word in freq_dict:
                    freq_dict[word] += 1
                else:
                    freq_dict[word] = 1
            temp={'id':i,'freq':freq_dict}
            freq_list.append(temp)
        return freq_list

    # Calculates the term frequency of words in each sentence
    def tf(self,text,freq_list):
        tf_list=[]
        for freq in freq_list:
            id=freq['id']
            for k in freq['freq']: # loop for each word
                # tf=count of word/total words in the sentence
                temp={'id':id,
                    'word':k,
                    'tf':freq['freq'][k]/text[id]['word_count']}
                tf_list.append(temp)
        return tf_list
    
    # Calculates the inverse document frequency of words in the text
    def idf(self,text,freq_list):
        idf_list=[]
        for freq in freq_list:# loop for each sentence
            id=freq['id']
            for k in freq['freq']: # loop for each word
                # idf=log(total sentences/count of sentences containing the word+1)
                count=sum([k in f['freq'] for f in freq_list]) # count of sentences containing the word
                temp={'id':id,'word':k,'idf':math.log(len(text)/(count+1))} ## +1 for smoothing
                idf_list.append(temp)
        return idf_list
    
    # Calculate the tf-idf of words in the text
    def calc_tf_idf(self,tf_list,idf_list):
        tf_idf_list=[]
        for tf in tf_list:
            for idf in idf_list:
                # if the word and sentence id are the same in both lists then calculate the tf-idf
                if(tf['id']==idf['id'] and tf['word']==idf['word']):
                    temp={'id':tf['id'],'word':tf['word'],'tf_idf':tf['tf']*idf['idf']}
                    tf_idf_list.append(temp)
        return tf_idf_list
    
    # Score the sentences based on the tf-idf of the words
    def score_sentences(self,sentences,tf_idf_list,text):
        sent_data=[]
        # loop for each sentence
        for txt in text:
            id=txt['id'] # get the sentence id
            score=0 # initialize the score
            for tf_idf in tf_idf_list:
                if(tf_idf['id']==id):
                    score+=tf_idf['tf_idf']
            # store the sentence id, score and the sentence
            temp={'id':id,'score':score,'sentence':sentences[id]}
            sent_data.append(temp)
        return sent_data
    
    # Rank the sentences based on the score
    def rank_sentence(self, data):
        count=0
        summary=[]
        # calculate the average score of the sentences
        for t_dict in data:
            count+=t_dict['score']
        avg=count/len(data) # average score
        # loop for each sentence
        for sent in data:
            if(sent['score']>=(avg*self.threshold)):
                summary.append(sent['sentence'])
        
        summary=" ".join(summary)
        return summary

    # Generate the summary
    def summary(self,text_input):
        # step1: clean the text
        sentences=self.clean_text(text_input)
        # step2: count the number of words in the text
        text=self.count_in_sentences(sentences)
        # step3: create a frequency dictionary for each word of the sentence
        freq_list=self.freq_dict(sentences)
        # step4: calculate the term frequency of words in each sentence
        tf_list=self.tf(text,freq_list)
        # step5: calculate the inverse document frequency of words in the text
        idf_list=self.idf(text,freq_list)
        # step6: calculate the tf-idf of words in the text
        tf_idf_list=self.calc_tf_idf(tf_list,idf_list)
        # step7: score the sentences based on the tf-idf of the words
        sent_data=self.score_sentences(sentences,tf_idf_list,text)
        # step8: rank the sentences based on the score
        summary=self.rank_sentence(sent_data)
        # step9: capitalize the first letter of the summary
        summary=self.preprocess.capitalize(summary)
        return summary
    
# if(__name__ == "__main__"):
#     summ=MachineLearningSummarization()
#     print(summ.summary("The cat isn't in the box. The cat likes the box. The box is in the house. The house is in the city. The city is in the country. The country is in the world."))