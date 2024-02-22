import re
import pandas as pd
import math

class DialogDataSet:
    dataset_path=""
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    def remove_emojis(self,text):
        pattern = r"\([^A-Za-z\s]+\)"

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

    def expand_contractions(self,text):
        # Define a dictionary of common contractions and their expanded forms
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"\'em", " them", text)
        text = re.sub(r"ma’am", "madam", text)
        short_forms = [
            ("btw", "by the way"),
            ("eg", "for example"),
            ("etc", "et cetera"),
            ("i.e.", "that is"),
            ("imho", "in my humble opinion"),
            ("lol", "laugh out loud"),
            ("msg", "message"),
            ("nsfw", "not safe for work"),
            ("omg", "oh my god"),
            ("plz", "please"),
            ("qr", "quick reply"),
            ("tbh", "to be honest"),
            ("wtf", "what the f*ck"),
        ]
        # Expand all short forms in the text.
        for short_form, expansion in short_forms:
            text = re.sub(r"\b{}\b".format(short_form), expansion, text)
        return text

    def check_file_pattern(self,text):
        # Define a regular expression pattern to match <file_...>
        pattern = r'<file_\w+>'
        
        # Search for the pattern in the text
        match = re.search(pattern, text)
        
        # If a match is found, return an empty string, otherwise return the original text
        return match

    def replace_repeated_chars(self,text):
        # Replace consecutive occurrences of ',' with a single ','
        text = re.sub(r'\,{2,}', ',', text)
        # Replace consecutive occurrences of '!' with a single '!'
        text = re.sub(r'!{2,}', '!', text)
        # Replace consecutive occurrences of '.' with a single '.'
        text = re.sub(r'\.{2,}', '.', text)
        # Replace consecutive occurrences of '?' with a single '?'
        text = re.sub(r'\?{2,}', '?', text)
        return text
    
    def create_conversation(self,text):
        # Initialize a dictionary to store messages for each speaker
        speaker_messages = []
        
        # Iterate through each line
        current_speaker = None
        prev_speaker=None
        temp_text=''
        for line in text:
            if(line==""):
                continue
            # Check if the line contains a speaker name
            match = re.match(r'^([A-Za-z\.\s_-]+):', line)
            if match:
                current_speaker = match.group(1)
            if prev_speaker != None and current_speaker!=prev_speaker:
                speaker_messages.append(temp_text)
            print(line)
            # Append the message to the current speaker's messages list
            if current_speaker==prev_speaker:
                temp_text = temp_text+ " "+line[len(current_speaker)+1:].strip()
            else:
                temp_text= line
            prev_speaker=current_speaker
        return speaker_messages

    def preprocess_text(self,text,isDialogue=False):
        print("text :", type(text))
        if(not isinstance(text, str) or self.check_file_pattern(text)):
            return ""
        # Remove non-English characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text=text.split("\n")
        if(isDialogue):
            text=self.create_conversation(text)
        # Remove leading and trailing whitespaces
        text=[re.sub(r"\s+", " ", self.expand_contractions(self.replace_repeated_chars(self.remove_emoticons(self.remove_emojis(statement)))).lower()) for statement in text ]
        text="<s></s>".join(text)
        # Add <s> and </s> to the beginning and end of the text
        text="<s>"+text+"</s>"
        return text

    def preprocess_data(self):
        data=pd.read_csv(self.dataset_path)
        for index,row in data.iterrows():
            print(index,'\n')
            row['dialogue']=self.preprocess_text(row['dialogue'],True)
            row['summary']=self.preprocess_text(row['summary'])
        data_cleaned =data.drop(data[(data == '').any(axis=1)].index)
        data_cleaned.to_csv('preprocessed_dataset.csv', index=False)


# test
if __name__ == '__main__':
    dataset = DialogDataSet("data/raw/summarization/samsum/train.csv")
    dataset.preprocess_data()