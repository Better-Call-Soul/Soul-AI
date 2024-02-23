# from summarization import MachineLearningSummarization
# summ=MachineLearningSummarization()
# print(summ.summary("The cat isn't in the box?.... The cat likes the box, The box is in the house. The house is in the city. The city is in the country. The country is in the world."))
from preprocessing import Preprocessing
text = "I'm going to store . I'll be back soon. I'm going to the store. I'll be back soon."
p = Preprocessing()
print(p.preprocess_text([text]))