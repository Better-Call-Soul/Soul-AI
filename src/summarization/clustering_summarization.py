from preprocess.capitalize import Capitalize
from preprocess.preprocess import Preprocessor
from preprocess.fastcoref import Fastcoref
from vectorizers.mini_lm import MiniLM
import re
from vectorizers.tf_idf_vectorizer import TFIDFVectorizer
from utils.utils import best_len_of_summary
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class ClusteringSummarization:
    def __init__(self):
        self.cluster_count = 3
        self.fastcoref=Fastcoref()
        self.capitalize=Capitalize()
        self.preprocess=Preprocessor()
        self.vectorize = MiniLM()
        self.org_sentences=[]
        self.clean_sentences=[]
        # self.length_of_summary=0
        self.vectorizer=None
        self.vectorizer = TFIDFVectorizer()

    
    def initialize(self):
        self.cluster_count=best_len_of_summary(self.org_sentences)
        
    # process the text data and create the nodes
    def process(self, text:str):
        text=self.fastcoref.coreference_resolution(text)
        # split the text into sentences and preprocess each sentence
        statements = re.findall(r'[^.!?]+[.!?]', text)
        # use a set to track seen statements and a list to store unique statements
        seen_statements = set()
        unique_statements = []

        for statement in statements:
            # remove leading/trailing whitespace and convert to lowercase for comparison
            clean_statement = statement.strip().lower()

            # if the statement is not already seen, add to unique list and seen set
            if clean_statement not in seen_statements:
                unique_statements.append(statement.strip())
                seen_statements.add(clean_statement)
        statements=unique_statements
        # save original statements to return it in the final summary
        self.org_sentences =  [statement.strip() for statement in statements]
        # preprocessing steps:
        # print("statements",statements)
        for sentence in statements:
            sentence=self.preprocess.clean(sentence,
            ["lower_sentence","remove_emojis","remove_emoticons","remove_nonascii_diacritic",
            "remove_emails","clean_html",
            "remove_url","replace_repeated_chars","expand_sentence","remove_non_alphabetic",
            "remove_extra_space","tokenize_sentence","check_sentence_spelling","detokenize_sentence"]
            ,"")[0]
            self.clean_sentences.append(sentence)
            
    def encode(self, sentences:list[str])->list[list[float]]:
        return self.vectorizer.fit_transform(sentences)
        
        
    def cluster(self,encoded_data:list[list[float]]):
        kmeans = KMeans(n_clusters=self.cluster_count)
        kmeans.fit(encoded_data)
        self.clusters = kmeans.labels_.tolist()
        print(self.clusters)
    
    def create_sentence_dictionary(self):
        self.sentenceDictionary = {}
        for idx, sentence in enumerate(self.org_sentences):
            # print(idx)
            self.sentenceDictionary[idx] = {
                'text': sentence,
                'cluster': self.clusters[idx],
                'cleaned': self.clean_sentences[idx]
            }
    
    def create_cluster_dictionary(self):
        self.clusterDictionary = {}
        for key, sentence in  self.sentenceDictionary.items():
            if sentence['cluster'] not in  self.clusterDictionary:
                self.clusterDictionary[sentence['cluster']] = []
            self.clusterDictionary[sentence['cluster']].append(sentence['cleaned'])
            sentence['idx'] = len(self.clusterDictionary[sentence['cluster']]) - 1
    
    def rank_sentences(self):
        self.maxCosineScores = {}
        for key, clusterSentences in self.clusterDictionary.items():
            self.maxCosineScores[key] = {}
            self.maxCosineScores[key]['score'] = -1
            tfidf_matrix = self.encode(clusterSentences)
            # print("tfidf_matrix",tfidf_matrix)

            cos_sim_matrix = cosine_similarity(tfidf_matrix)
            # print("cos_sim_matrix",cos_sim_matrix)
            for idx, row in enumerate(cos_sim_matrix):
                sum = 0
                for col in row:
                    sum += col
                if sum >  self.maxCosineScores[key]['score']:
                    self.maxCosineScores[key]['score'] = sum
                    self.maxCosineScores[key]['idx'] = idx

    def summary(self, text:str)->str:
        # process the text data
        self.process(text)
        
        # intialize the cluster count
        self.initialize()
        # encode the text data
        encoded_data=self.encode(self.clean_sentences)
        
        # cluster the data
        self.cluster(encoded_data)
        
        # create a dictionary of sentences
        self.create_sentence_dictionary()
        
        # create a dictionary of clusters
        self.create_cluster_dictionary()
        # rank the sentences
        self.rank_sentences()
        resultIndices = []
        i = 0
        for key, value in self.maxCosineScores.items():
            cluster = key
            idx = value['idx']
            for key, value in self.sentenceDictionary.items():
                if value['cluster'] == cluster and value['idx'] == idx:
                    resultIndices.append(key)

        resultIndices.sort()
        
        output_summary = ''
        for idx in resultIndices:
            output_summary += self.org_sentences[idx] + ' '
            
        return output_summary