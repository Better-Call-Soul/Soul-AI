import re
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from preprocess.preprocess import Preprocessor
from preprocess.fastcoref import Fastcoref
from preprocess.capitalize import Capitalize

# Class for machine learning based summarization of text using tf-idf algorithm
class TfIdfSummarization:
    def __init__(self,threshold=0.9):
        self.threshold=threshold
        self.preprocess=Preprocessor()
        self.fastcoref=Fastcoref()
        self.capitalize=Capitalize()
        self.originalStatements = None
    # Clean the text
    def clean_text(self,text:str) -> list[str]:
        text=self.fastcoref.coreference_resolution(text)
        statements = re.findall(r'[^.!?]+[.!?]', text)
        self.originalStatements = statements
        sentences =[]
        # preprocessing steps:
        for sentence in statements:
            sentence=self.preprocess.clean(sentence,
            ["lower_sentence","remove_emojis","remove_emoticons","remove_nonascii_diacritic","remove_emails","clean_html",
            "remove_url","replace_repeated_chars","expand_sentence","remove_extra_space","tokenize_sentence","check_sentence_spelling","detokenize_sentence"]
            ,"")[0]
            sentences.append(sentence)

        return sentences
    
    # Count the number of words in the text
    def count_words(self,text:str) -> int:
        # Tokenize the text
        words = word_tokenize(text)
        words=[word for word in words if word not in  stopwords.words('english')] # remove non-alphabetic characters
        return len(words)
    
    # Count the number of words in each sentence
    def count_in_sentences(self, sentences:list[str]) -> list[dict[str,int]]:
        count=[]
        for i,sentence in enumerate(sentences):
            temp={'id':i,'word_count':self.count_words(sentence[:-1])}
            count.append(temp)
        return count
    

    
    # Create a frequency dictionary for each word of the sentence
    def freq_dict(self,sentences : list[str])  -> list[dict[int,dict[str,int]]]:
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
    def tf(self,text,freq_list : list[dict[int,dict[str,int]]]) -> list[dict[int,str,float]]:
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
    def idf(self,text,freq_list: list[dict[int,dict[str,int]]]) -> list[dict[int,str,float]]:
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
    def calc_tf_idf(self,tf_list,idf_list : list[dict[int,str,float]]) -> list[dict[int,str,float]]:
        tf_idf_list=[]
        for tf in tf_list:
            for idf in idf_list:
                # if the word and sentence id are the same in both lists then calculate the tf-idf
                if(tf['id']==idf['id'] and tf['word']==idf['word']):
                    temp={'id':tf['id'],'word':tf['word'],'tf_idf':tf['tf']*idf['idf']}
                    tf_idf_list.append(temp)
        return tf_idf_list
    
    # Score the sentences based on the tf-idf of the words
    def score_sentences(self,sentences: list[str],tf_idf_list:list[dict[int,str,float]] ,text : list[dict[int,str,int]]) -> list[dict[int,float,str]]:
        sent_data=[]
        # loop for each sentence
        for txt in text:
            id=txt['id'] # get the sentence id
            score=0 # initialize the score
            for tf_idf in tf_idf_list:
                if(tf_idf['id']==id):
                    score+=tf_idf['tf_idf']
            # store the sentence id, score and the sentence
            temp={'id':id,'score':score,'sentence':sentences[id],
                "original_sentence":self.originalStatements[id].strip()}
            sent_data.append(temp)
        return sent_data

    # Calculate the best number of sentences for the summary
    def best_num_sentences(self,sentences: list[str]) -> int:
        # if the number of sentences is less than or equal to 3 then return the number of sentences
        if len(sentences) <= 3:
            return len(sentences)
        # else return 1.3 times the log of the number of sentences
        return round(1.3 * math.log(len(sentences)))
    
    # Rank the sentences based on the score
    def rank_sentence_by_avg(self, data: list[dict[int,float,str]]) -> str:
        count=0
        summary=[]
        # calculate the average score of the sentences
        for t_dict in data:
            count+=t_dict['score']
        avg=count/len(data) # average score
        threshold_score = avg * self.threshold
        # loop for each sentence
        summary = [item['original_sentence'] for item in data if item['score'] >= threshold_score]

        summary=" ".join(summary)
        return summary

    # Get the top sentences based on the score and limit by the best number of sentences
    def get_top_sentences(self,data: list[dict[int,float,str]]) -> str:
        # sort the sentences based on the score
        data.sort(key=lambda x: x['score'], reverse=True)
        # get the best number of sentences for the summary
        best_num=self.best_num_sentences(data)
        # get the top sentences
        top_sentences=data[:best_num]
        # loop for each sentence
        summary = [item['original_sentence'] for item in top_sentences]
        
        summary=" ".join(summary)
        return summary
    
    # Generate the summary
    def summary(self,text_input: str) -> str:
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
        # summary=self.rank_sentence_by_avg(sent_data)
        summary=self.get_top_sentences(sent_data)
        # step9: capitalize the first letter of the summary
        summary=self.capitalize.capitalize(summary)
        return summary
    
# if(__name__ == "__main__"):
#     summ=MachineLearningSummarization()
#     print(summ.summary("The cat isn't in the box. The cat likes the box. The box is in the house. The house is in the city. The city is in the country. The country is in the world."))