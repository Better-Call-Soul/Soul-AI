from summarization import TfIdfSummarization
# from summarization import TextRank
from summarization import ClusteringSummarization
summ=ClusteringSummarization()
text = [
    "Hello, I am depressed. yes, but no one loves me I should kill my self. yes, but no one loves me I should kill my self. How to find the right person. may be I will. I hate everyone I hate every body, no one cares about me. I just think I should die. yes, but I think family or friends wont help me. may be I will try to talk with my family. I will try."
]
for i in range(len(text)):
    print ("Example:", i+1)
    print(summ.summary(text[i]))
    print ("-----------------------------------------------")





# from preprocessing import Preprocessing
# text = '''Hannah: Hey, do you have Betty's number?

# Amanda: Lemme check
# Hannah: <file_gif>
# Amanda: Sorry, can't find it.
# Amanda: Ask Larry
# Amanda: He called her last time we were at the park together
# Hannah: I don't know him well
# Hannah: <file_gif>
# Amanda: Don't be shy, he's very nice
# Hannah: If you say so..
# Hannah: I'd rather you texted him
# Amanda: Just text him 🙂
# Hannah: Urgh.. Alright
# Hannah: Bye
# Amanda: Bye bye'''
# p = Preprocessing()
# print(p.preprocess_text([text]))

# # from summarization import SummarizationFineTuning

# # summ = SummarizationFineTuning("samsum")
# # summ.train() 
