import spacy
import re


## Class for capitalizing the entities in the text using spaCy NER (Named Entity Recognition) model
class Capitalize:
    def __init__(self):
        # Load the English language model
        self.nlp = spacy.load("en_core_web_sm")
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