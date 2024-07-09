

from fastcoref import spacy_component
import spacy
import re

# Class for coreference resolution using fastcoref model
class Fastcoref:
    def __init__(self):
        # self.nlp_fastcoref = spacy.load("en_core_web_sm")
        # self.nlp_fastcoref.add_pipe("fastcoref")
        self.nlp_fastcoref = spacy.load("../data/embeddings/summarization/fastcoref")
   
    def coreference_resolution(self,text:str)->str:
        doc =self.nlp_fastcoref(      # for multiple texts use nlp.pipe
        text,
        component_cfg={"fastcoref": {'resolve_text': True}}
        )
        return doc._.resolved_text