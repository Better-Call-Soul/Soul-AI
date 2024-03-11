# from summarization import MachineLearningSummarization
# summ=MachineLearningSummarization()
# print(summ.summary("Not great, to be honest. User've been feeling very low and hopeless lately. Yes, but User don't see any point in living anymore. User just feel no one loves User. what should User do? User don't believe Assistant. Nothing ever changes for User. User've tried everything, and nothing works. That is maybe good. Thank Assistant. User don't know. User guess User can try. User have nothing to lose. Haha right. Mmmm User like to go clubbing and swim. Yes, swimming changes User's mood. Thank Assistant, goodbye."))
from preprocessing import Preprocessing
text = "supreme court to go paperless in 6 months: cji"
p = Preprocessing()
print(p.preprocess_text([text]))