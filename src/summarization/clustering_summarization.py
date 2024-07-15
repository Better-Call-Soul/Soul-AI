from preprocess.capitalize import Capitalize
from preprocess.preprocess import Preprocessor
from preprocess.fastcoref import Fastcoref
from vectorizers.mini_lm import MiniLM
import re
from vectorizers.tf_idf_vectorizer import TFIDFVectorizer
from utils.utils import best_len_of_summary
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import Literal
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import numpy as np


class ClusteringSummarization:
    def __init__(self,embeddings_type: Literal['tf-idf', 'mini-lm']='mini-lm',
                cluster_type: Literal['kmeans', 'hierarchy','dbscan']='hierarchy'):
        self.n_centroids = 0
        self.fastcoref=Fastcoref()
        self.capitalize=Capitalize()
        self.preprocess=Preprocessor()
        self.org_sentences=[]
        self.clean_sentences=[]
        self.embeddings_type=embeddings_type
        self.cluster_type=cluster_type
        # check the embeddings type
        if(embeddings_type=='tf-idf'):
            self.vectorizer = TFIDFVectorizer()
        elif(embeddings_type=='mini-lm'):
            self.vectorizer = MiniLM()
        else:
            raise ValueError("Invalid embeddings type")

    
    def initialize(self):
        self.n_centroids=best_len_of_summary(self.org_sentences)
        
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
            "remove_extra_space","tokenize_sentence","remove_stop_words","lemm_sentence",
            "check_sentence_spelling","detokenize_sentence"]
            ,"")[0]
            self.clean_sentences.append(sentence)
    
    # encode the text data
    def encode(self, sentences:list[str])->list[list[float]]:
        if (self.embeddings_type=='mini-lm'):
            return self.vectorizer.encode(sentences)
        else:
            return self.vectorizer.fit_transform(sentences)
        
    # cluster the data
    def cluster(self,encoded_data:list[list[float]]):
        #############  Kmeans ###############
        if self.cluster_type=='kmeans':
            kmeans = KMeans(n_clusters=self.n_centroids)
            kmeans.fit(encoded_data)
            self.clusters = kmeans.labels_.tolist()
        #############  Hierarchy ###############
        elif self.cluster_type=='hierarchy':
            # linkage matrix
            Z = linkage(encoded_data, 'ward')
            # get the distance between the clusters being merged.
            distances = Z[:, 2]
            # print("len ",(distances))
            # calculate mean and standard deviation of the distances
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)

            # set max_d dynamically (e.g., mean + 1/2* standard deviation)
            max_d = mean_distance + std_distance/2
            # cluster the data
            self.clusters = fcluster(Z, max_d, criterion='distance')
        #############  DBSCAN ###############
        elif self.cluster_type=='dbscan':
            distances = pairwise_distances(encoded_data)
            eps = max(np.mean(distances),1)
            # print("eps",eps)
            min_samples = max(int(len(encoded_data) * 0.1),1)
            # print("min_samples",min_samples)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            self.clusters = dbscan.fit_predict(encoded_data)
        else:
            raise ValueError("Invalid cluster type")
        # print(self.clusters)
    
    # create a dictionary of sentences
    def create_sentence_dictionary(self):
        self.sentenceDict = {}
        # loop for each sentence
        for key, sentence in enumerate(self.org_sentences):
            # print(idx)
            # fill the dictionary with the sentence, cleaned sentence and cluster
            self.sentenceDict[key] = {
                'text': sentence,
                'cleaned': self.clean_sentences[key],
                'cluster': self.clusters[key],
            }
    
    # create a dictionary of clusters
    def create_cluster_dictionary(self):
        self.clustersDic = {}
        # loop for each sentence in the sentence dictionary
        for _, sentence in  self.sentenceDict.items():
            # if the cluster is not in the dictionary, add it
            if sentence['cluster'] not in  self.clustersDic:
                # add the cluster to the dictionary
                self.clustersDic[sentence['cluster']] = []
            # append the cleaned sentence to the cluster
            self.clustersDic[sentence['cluster']].append(sentence['cleaned'])
            sentence['key'] = len(self.clustersDic[sentence['cluster']]) - 1
    
    # rank the sentences
    def rank_sentences(self):
        self.cosineScore = {}
        # loop for each cluster in the cluster dictionary
        for key, clusterSentences in self.clustersDic.items():
            self.cosineScore[key] = {}
            # initialize the score to -1 for each cluster
            self.cosineScore[key]['score'] = -1
            # encode the cluster sentences 
            matrix = self.encode(clusterSentences)
            # calculate the cosine similarity matrix
            cos_sim_matrix = cosine_similarity(matrix)
            # loop for each row in the cosine similarity matrix
            for index, row in enumerate(cos_sim_matrix):
                # calculate the sum of the row
                score=sum(row)
                # if the score is greater than the previous score, update the score and the key
                if score >  self.cosineScore[key]['score']:
                    self.cosineScore[key]['score'] = score
                    self.cosineScore[key]['key'] = index

    # generate the summary of the text
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
        outputIndices = []
        # loop for each cluster in the cosine score dictionary
        for cluster, obj in self.cosineScore.items():
            # get the index of the sentence
            index = obj['key']
            # loop for each sentence in the sentence dictionary
            for index_statement, sentence_obj in self.sentenceDict.items():
                if sentence_obj['cluster'] == cluster and sentence_obj['key'] == index:
                    outputIndices.append(index_statement)
                
        # sort the indices of the sentences in the output to be in order
        outputIndices.sort()
        
        # create the output summary
        output_summary = ''
        for idx in outputIndices:
            output_summary += self.org_sentences[idx] + ' '
        # capitalize the first letter of the summary
        output_summary = self.capitalize.capitalize(output_summary)
        return output_summary