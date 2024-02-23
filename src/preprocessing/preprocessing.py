
import re
from typing import List
from utils.expand_contractions import *
import spacy
from fastcoref import spacy_component

# class for preprocessing the text data and soma basic NLP tasks
class Preprocessing:
    def __init__(self):
        self.contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
        # Load the English language model
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp_fastcoref = spacy.load("en_core_web_sm")
        self.nlp_fastcoref.add_pipe("fastcoref")
    def remove_emojis(self,text:str) -> str:
        pattern = r"\([^A-Za-z]*\)"
        # Remove the pattern from the text using the regular expression
        text = re.sub(pattern, '', text)
        emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002500-\U00002BEF"  # chinese char
                u"\U00002700-\U000027BF"  # Dingbats
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u"\U00010000-\U0010ffff"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u200d"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        return text

    def remove_emoticons(self,text):
        # Define a regular expression pattern to match emoticons
        emoticon_pattern = re.compile(r':(\)+)|:-(\))+|;(\))+|:-(D)+|:(D)+|;-(D)+|x(D)+|X(D)+|:-(\()+|:(\()+|:-(/)+|:(/)+|:-(\))+||:(\))+||:-(O)+|:(O)+|:-(\*)+|:(\*)+|<(3)+|:(P)+|:-(P)+|;(P)+|;-(P)+|:(S)+|>:(O)+|8(\))+|B-(\))+|O:(\))+', flags=re.IGNORECASE)
        # Remove emoticons using the pattern
        return emoticon_pattern.sub('', text)

    # Expand the contractions in the text
    def expand_contractions(self,text:str) -> str:
        def replace(match):
            return contractions_dict[match.group(0)]
        # Replace the contractions in the text
        return self.contractions_re.sub(replace, text)

    def replace_repeated_chars(self,text:str) -> str:
        # Replace consecutive occurrences of ',' with a single ','
        text = re.sub(r'\,{2,}', ',', text)
        # Replace consecutive occurrences of '!' with a single '!'
        text = re.sub(r'!{2,}', '!', text)
        # Replace consecutive occurrences of '.' with a single '.'
        text = re.sub(r'\.{2,}', '.', text)
        # Replace consecutive occurrences of '?' with a single '?'
        text = re.sub(r'\?{2,}', '?', text)
        statements = re.findall(r'[^.!?,]+[.!?,]', text)
        # Remove any leading or trailing whitespace from each statement
        text = [statement.strip() for statement in statements]
        text = " ".join(text)
        return text
    
    # Function to clean the html from the article
    def clean_HTML(self,text:str) -> str:
        cleanr = re.compile('<.*?>')
        text = re.sub(cleanr, '', text)
        return text

    def clean_text(self,text:str) -> str:
        text=  text.lower()
        # Removing the email ids
        text =  re.sub('\S+@\S+','', text)
        
        # Removing the contractions
        text =  self.expand_contractions(text)
        # Removing The URLS
        text =  re.sub("((http\://|https\://|ftp\://)|(www.))+(([a-zA-Z0-9\.-]+\.[a-zA-Z]{2,4})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(/[a-zA-Z0-9%:/-_\?\.'~]*)?",'', text)

        # Stripping the possessives
        text =  text.replace("'s", '')
        text =  text.replace('’s', '')
        text =  text.replace("\'s", '')
        text =  text.replace("\’s", '')

        # Removing the Trailing and leading whitespace and double spaces
        text =  re.sub(' +', ' ',text)

        # Removing the Trailing and leading whitespace and double spaces again as removing punctuation might
        # Lead to a white space
        text =  re.sub(' +', ' ',text)

        return text
        
    def preprocess_text(self,text:List[str]) -> List[str]:
        # Remove non-English characters
        text = [re.sub(r'[^\x00-\x7F]+', '', statement)for statement in text]
        # Remove leading and trailing whitespaces
        text=[re.sub(r"\s+", " ", self.clean_HTML( # Expand the contractions in the text
                                self.replace_repeated_chars( # Replace repeated characters in the text
                                self.remove_emoticons( # Remove emoticons from the text
                                self.remove_emojis(
                                self.clean_text(statement.lower())))))) # Clean the text
                                for statement in text ]
        return text

    # Function to capitalize the entities in the text
    def capitalize(self,text:str) -> str:
        # Process the text using spaCy
        doc = self.nlp(text)

        # Iterate over each entity in the document
        for ent in doc.ents:
            # Check if the entity is a GPE (Geopolitical Entity, like cities)
            if ent.label_ in ("GPE", "PERSON", "ORG"):
                # Capitalize the entity text
                text = text.replace(ent.text, ent.text.capitalize())
        def uppercase(matchobj):
            return matchobj.group(0).upper()
        # 1) Capitalize the first of the string if it is a letter
        # 2) Capitalize after every period, question mark or exclamation mark
        # 3) Capitalize the letter if there is a period after it and no letters before it
        
        text=re.sub('^([a-z])|[\.|\?|\!]\s*([a-z])|\s+([a-z])(?=\.)', uppercase, text)
        return text
    def coreference_resolution(self,text:str)->str:
        doc =self.nlp_fastcoref(      # for multiple texts use nlp.pipe
        text,
        component_cfg={"fastcoref": {'resolve_text': True}}
        )
        return doc._.resolved_text

# if __name__ == "__main__":
#     text = "I'm going to the store . I'll be back soon. I'm going to the store. I'll be back soon."
#     p = Preprocessing()
#     print(p.preprocess_text([text]))