from transformers import pipeline

# Classifier class
# This class is a wrapper for the GoEmotions model from the Hugging Face Transformers library
# example predict: [{'label': 'admiration', 'score': 0.9999999403953552}.....]

# output labels:
# admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire
# disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief
# joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

class Classifier:
    def __init__(self, model_path="SamLowe/roberta-base-go_emotions"):
        # load the GoEmotions model
        self.model = pipeline(task="text-classification", model=model_path, top_k=None)

    # predict the emotions in the text
    def predict(self, text:list[str])->object:
        return self.model(text)
    
    # calculate the mental health score
    def mental_health_score(self,emotions):
        # define a set of labels associated with mental health concerns
        mental_health_labels = {
            'sadness', 'disappointment', 'annoyance', 'disapproval',
            'nervousness', 'remorse', 'embarrassment', 'anger',
            'disgust', 'grief', 'confusion', 'fear'
        }

        # initialize variables to calculate the score
        total_score = 0
        mental_score = 0

        # Iterate over the emotions
        for emotion in emotions:
            label = emotion['label']
            score = emotion['score']
            if label in mental_health_labels:
                mental_score += score
            total_score += score

        # normalize score
        normalized_score = mental_score / total_score

        return normalized_score
