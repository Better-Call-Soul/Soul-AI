
import re
from typing import List
import contractions

# class for preprocessing the text data and soma basic NLP tasks
class Preprocessing:
    def __init__(self):
        pass

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
        return contractions.fix(text)

    def replace_repeated_chars(self,text:str) -> str:
        # Replace consecutive occurrences of ',' with a single ','
        text = re.sub(r'\,{2,}', ',', text)
        # Replace consecutive occurrences of '!' with a single '!'
        text = re.sub(r'!{2,}', '!', text)
        # Replace consecutive occurrences of '.' with a single '.'
        text = re.sub(r'\.{2,}', '.', text)
        # Replace consecutive occurrences of '?' with a single '?'
        text = re.sub(r'\?{2,}', '?', text)

        return text
    
    # Function to clean the html from the article
    def clean_HTML(self,text:str) -> str:
        cleanr = re.compile('<.*?>')
        text = re.sub(cleanr, '', text)
        return text

    def clean_text(self,text:str) -> str:
        # Removing the email ids
        text =  re.sub('\S+@\S+','', text)
        
        # Removing the contractions
        text =  self.expand_contractions(text)
        # Removing The URLS
        text =  re.sub("((http\://|https\://|ftp\://)|(www.))+(([a-zA-Z0-9\.-]+\.[a-zA-Z]{2,4})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(/[a-zA-Z0-9%:/-_\?\.'~]*)?",'', text)

        # Removing the Trailing and leading whitespace and double spaces
        text =  re.sub(' +', ' ',text)

        # Removing the Trailing and leading whitespace and double spaces again as removing punctuation might
        # Lead to a white space
        text =  re.sub(' +', ' ',text)
        # remove spaces
        text = re.sub("[ \t\r]+", " ", text)
        # remove nonenglish letter
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text

    def preprocess_text(self,text:List[str]) -> List[str]:
        print(text)
        # Remove non-English characters
        # text=[statement.strip() for statement in text]
        # Remove leading and trailing whitespaces
        text=[ self.clean_HTML( # Expand the contractions in the text
                                self.replace_repeated_chars( # Replace repeated characters in the text
                                self.remove_emoticons( # Remove emoticons from the text
                                self.remove_emojis(
                                self.clean_text(statement.strip()))))) # Clean the text
                                for statement in text ]
        return text



# if __name__ == "__main__":
    # text = "4 ex-bank officials booked for cheating bank of ₹209 crore"
    # p = Preprocessing()
    # print(p.preprocess_text([text]))