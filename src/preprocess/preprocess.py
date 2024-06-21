import nltk
from nltk.data import find
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import contractions
import unicodedata
from bs4 import BeautifulSoup
import emoji
import re
from spellchecker import SpellChecker
import unittest

class Preprocessor:
    
    resources = [
        'tokenizers/punkt',
        'corpora/stopwords',
        'corpora/wordnet',
        'taggers/averaged_perceptron_tagger'
    ]

    def __init__(self) -> None:
        for resource in self.resources:
            try:
                find(resource)
                print(f"{resource} is already downloaded.")
            except LookupError:
                print(f"{resource} not found. Downloading...")
                download(resource.split('/')[1])

        # Stopword removal
        self.stop_words = set(stopwords.words('english'))
        # Initialize the WordNet lemmatizer
        self.lemmatizer = WordNetLemmatizer()

    def lower_sentence(self, sentence: str) -> str:
        '''
        Lowercase the sentence.
        :param data: The sentence to lowercase.
        :return: The lowercased sentence
        :rtype: str
        '''
        return sentence.lower()
    
    def remove_emails(self, sentence: str) -> str:
        '''
        Remove emails from the sentence.
        :param sentence: The sentence to remove emails from.
        :type sentence: str
        :return: The sentence without emails.
        :rtype: str
        '''
        return re.sub(r"\S*@\S*\s?", "", sentence)
    
    def remove_nonascii_diacritic(self, sentence: str) -> str:
        '''

        Remove diacritics from the sentence.

        :param sentence: The sentence to remove diacritics from.

        :type sentence: str

        :return: The sentence without diacritics.

        :rtype: str
        '''

        return unicodedata.normalize("NFKD", sentence).encode("ascii", "ignore").decode("utf-8", "ignore")
    
    def clean_html(self, sentence: str) -> str:
        '''
        Remove HTML tags from the sentence.
        :param sentence: The sentence to remove HTML tags from.
        :type sentence: str
        :return: The sentence without HTML tags.
        :rtype: str
        '''
        return BeautifulSoup(sentence, "html.parser").get_text()

    def replace_repeated_chars(self, sentence: str) -> str:
        '''
        Replace repeated characters in the sentence.
        :param sentence: The sentence to replace repeated characters in.
        :type sentence: str
        :return: The sentence with replaced repeated characters.
        :rtype: str
        '''
        # Replace consecutive occurrences of ',', '!', '.', and '?' with a single occurrence
        return re.sub(r'([,!?.])\1+', r'\1', sentence)

    def translate_emojis_to_text(self, sentence: str) -> str:
        '''
        Translate emojis in the sentence to text.
        :param sentence: The sentence to translate emojis to text.
        :type sentence: str
        :return: The sentence with translated emojis to text.
        :rtype: str
        '''
        line = ''
        for char in sentence:
            if emoji.is_emoji(char):
                emoji_text = emoji.demojize(char)[1:-1].replace('_', ' ')
                line += emoji_text
            else:
                line += char
        
        return line

    def expand_sentence(self, sentence: str) -> str:
        '''
        Expand the contractions in the sentence.
        :param sentence: The sentence to expand contractions in.
        :type sentence: str
        :return: The sentence with expanded contractions.
        :rtype: str
        '''
        return contractions.fix(sentence)

    def remove_url(self, sentence: str) -> str:
        '''
        Remove URLs from the sentence.
        :param sentence: The sentence to remove URLs from.
        :type sentence: str
        :return: The sentence without URLs.
        :rtype: str
        '''
        return re.sub("((http\://|https\://|ftp\://)|(www.))+(([a-zA-Z0-9\.-]+\.[a-zA-Z]{2,4})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(/[a-zA-Z0-9%:/-_\?\.'~]*)?", '', sentence)

    def remove_possessives(self, sentence: str) -> str:
        '''
        Strip possessives from the sentence.
        :param sentence: The sentence to strip possessives from.
        :type sentence: str
        :return: The sentence without possessives.
        :rtype: str
        '''
        # Stripping the possessives
        sentence = sentence.replace("'s", '')
        sentence = sentence.replace('’s', '')
        sentence = sentence.replace('s’', 's')
        sentence = sentence.replace("s'", 's')
        return sentence

    def remove_extra_space(self, sentence: str) -> str:
        '''
        Remove extra spaces from the sentence.
        :param sentence: The sentence to remove extra spaces from.
        :type sentence: str
        :return: The sentence without extra spaces.
        :rtype: str
        '''
        return re.sub(r'\s+', ' ', sentence).strip()


    def check_sentence_spelling(self, sentence: list[str]) -> list[str]:
        '''
        Check the spelling of the words in the sentence.
        :param sentence: The sentence to check the spelling of.
        :type sentence: list
        :return: The sentence with corrected spelling.
        :rtype: list
        '''
        spell = SpellChecker()
        corrected_sentence = []
        for word in sentence:
            if word != '':
                correction = spell.correction(word)
                if correction is not None:
                    corrected_sentence.append(correction)
                else:
                    corrected_sentence.append(word)
            else:
                corrected_sentence.append('')
        return corrected_sentence

    def tokenize_sentence(self, sentence: str) -> list[str]:
        '''
        Tokenize the sentence.
        :param sentence: The sentence to tokenize.
        :type sentence: str
        :return: The tokenized sentence.
        :rtype: str
        '''
        return nltk.word_tokenize(sentence)
    

    def remove_stop_words(self, sentence: list[str]) -> list[str]:
        '''
        Remove stop words from the sentence.
        :param sentence: The sentence to remove stop words from.
        :type sentence: list[str]
        :return: The sentence without stop words.
        :rtype: list[str]
        '''
        return [word for word in sentence if word not in self.stop_words]

    def lemm_sentence(self, sentence: list[str]) -> list[str]:
        '''
        Lemmatize the sentence.
        :param sentence: The sentence to lemmatize.
        :type sentence: list[str]
        :return: The lemmatized sentence.
        :rtype: list[str]
        '''
        # Perform POS tagging
        pos_tags = pos_tag(sentence)
        # Lemmatize each word based on its POS tag
        lemmatized_words = []
        for word, pos in pos_tags:
            # Map Penn Treebank POS tags to WordNet POS tags
            if pos.startswith('N'):  # Nouns
                pos = 'n'
            elif pos.startswith('V'):  # Verbs
                pos = 'v'
            elif pos.startswith('J'):  # Adjectives
                pos = 'a'
            elif pos.startswith('R'):  # Adverbs
                pos = 'r'
            else:
                pos = 'n'  # Default to noun if POS tag not found

            # Lemmatize the word using the appropriate POS tag
            lemma = self.lemmatizer.lemmatize(word, pos=pos)
            lemmatized_words.append(lemma)
        return lemmatized_words
    
    def clean(self, line: str, steps: list[str] = None) -> list[str]:
        '''
        Clean the line and return it as a list of tokens
        :param line: the line to clean
        :type line: str
        :param steps: list of steps to apply
        :type steps: list[str]
        :return: the cleaned line as a list of tokens
        :rtype: list
        '''
        # Default steps to apply if none are specified
        default_steps = [
            'translate_emojis_to_text',
            'lower_sentence',
            'remove_nonascii_diacritic',
            'remove_emails',
            'clean_html',
            'remove_url',
            'replace_repeated_chars',
            'expand_sentence',
            'remove_possessives',
            'remove_extra_space',
            'tokenize_sentence',
            'check_sentence_spelling',
            'remove_stop_words',
            'lemm_sentence'
        ]
        
        # Use specified steps if provided, otherwise use default steps
        if steps is None:
            steps = default_steps

        # Define the processing functions
        processing_functions = {
            'translate_emojis_to_text': self.translate_emojis_to_text,
            'lower_sentence': self.lower_sentence,
            'remove_nonascii_diacritic': self.remove_nonascii_diacritic,
            'remove_emails': self.remove_emails,
            'clean_html': self.clean_html,
            'remove_url': self.remove_url,
            'replace_repeated_chars': self.replace_repeated_chars,
            'expand_sentence': self.expand_sentence,
            'remove_possessives': self.remove_possessives,
            'remove_extra_space': self.remove_extra_space,
            'tokenize_sentence': self.tokenize_sentence,
            'check_sentence_spelling': self.check_sentence_spelling,
            'remove_stop_words': self.remove_stop_words,
            'lemm_sentence': self.lemm_sentence
        }
        
        # Apply the specified steps
        for step in steps:
            if step in processing_functions:
                line = processing_functions[step](line)

        # Ensure tokenize_sentence was applied
        if isinstance(line, str):
            line = [line]

        if len(line) == 0:
            return ['Normal']
        
        return line

def test() -> None:
    class TestPreprocessor(unittest.TestCase):

        def setUp(self):
            self.preprocessor = Preprocessor()

        def test_lower_sentence(self):
            self.assertEqual(self.preprocessor.lower_sentence("HELLO WORLD"), "hello world")

        def test_remove_emails(self):
            self.assertEqual(self.preprocessor.remove_emails("Contact me at test@example.com"), "Contact me at ")

        def test_remove_nonascii_diacritic(self):
            self.assertEqual(self.preprocessor.remove_nonascii_diacritic("café"), "cafe")

        def test_clean_html(self):
            self.assertEqual(self.preprocessor.clean_html("<p>Hello, world!</p>"), "Hello, world!")

        def test_replace_repeated_chars(self):
            self.assertEqual(self.preprocessor.replace_repeated_chars("Heeellooo!!!!"), "Heeellooo!")

        def test_translate_emojis_to_text(self):
            self.assertEqual(self.preprocessor.translate_emojis_to_text("Hello 😊"), "Hello smiling face with smiling eyes")

        def test_expand_sentence(self):
            self.assertEqual(self.preprocessor.expand_sentence("can't won't"), "cannot will not")

        def test_remove_url(self):
            self.assertEqual(self.preprocessor.remove_url("Check http://example.com"), "Check ")

        def test_remove_possessives(self):
            self.assertEqual(self.preprocessor.remove_possessives("John's car"), "John car")

        def test_remove_extra_space(self):
            self.assertEqual(self.preprocessor.remove_extra_space("This  is   a test"), "This is a test")

        def test_tokenize_sentence(self):
            self.assertEqual(self.preprocessor.tokenize_sentence("This is a test."), ['This', 'is', 'a', 'test', '.'])

        def test_check_sentence_spelling(self):
            self.assertEqual(self.preprocessor.check_sentence_spelling(['This', 'is', 'a', 'tst']), ['This', 'is', 'a', 'test'])

        def test_remove_stop_words(self):
            self.assertEqual(self.preprocessor.remove_stop_words(['This', 'is', 'a', 'test']), ['This', 'test'])

        def test_lemm_sentence(self):
            self.assertEqual(self.preprocessor.lemm_sentence(['running', 'jumps', 'easily']), ['run', 'jump', 'easily'])

        def test_clean_with_default_steps(self):
            test_line = "This is a test line with an email@example.com and a link http://example.com 😊"
            cleaned_line = self.preprocessor.clean(test_line)
            self.assertEqual(cleaned_line, ['test', 'line', 'link', 'smile', 'face', 'smile', 'eye'])

        def test_clean_with_custom_steps(self):
            test_line = "This is a test line with an email@example.com and a url http://example.com"
            steps = ['lower_sentence', 'remove_emails', 'remove_url', 'tokenize_sentence']
            cleaned_line = self.preprocessor.clean(test_line, steps=steps)
            self.assertEqual(cleaned_line, ['this', 'is', 'a', 'test', 'line', 'with', 'an', 'and', 'a', 'url'])

    # Instantiate the test class and run it
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPreprocessor)
    unittest.TextTestRunner().run(suite)


if __name__ == '__main__':
    print('test 1: Running a simple test case...')
    preprocessor = Preprocessor()
    line = "This is a sample sentence."
    cleaned_line = preprocessor.clean(line)
    print(cleaned_line)
    print('test 2: Running The Unit test...')
    # Call the test function to run the tests
    test()
    print('Exit...')