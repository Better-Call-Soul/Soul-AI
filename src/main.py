from summarization import TfIdfSummarization
summ=TfIdfSummarization()
text = [
    """
    Today 😄 was one of those days where I was reminded of the perks of freelancing. I started my morning by taking a walk in the park. It’s a luxury to start the day with some fresh air and not rush to an office. When I got back home, I settled into my workspace and checked my schedule. I had two client calls in the morning, both went well, and it looks like I’ll be starting on a new project soon.

    I spent most of the afternoon working on a current project. It's a website redesign for a small boutique, and I'm really enjoying the creative freedom they’ve given me. I took a short break for lunch, just a sandwich and some fruit, and then it was back to work.

    In the evening, I joined a webinar on digital marketing trends. It's important to keep up with the industry, and these webinars are great for that. Afterwards, I made some dinner and then worked for another hour or two.

    I wrapped up my workday by responding to some emails and planning my tasks for tomorrow. To unwind, I played some guitar and watched a movie. I’m trying to make more time for hobbies, and it’s been great for my mental health.
    """
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
