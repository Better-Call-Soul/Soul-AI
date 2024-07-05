from summarization import TfIdfSummarization
summ=TfIdfSummarization()
text = [
    """
    Today was a typical whirlwind day for me. I woke up around 6:30 AM, which is a bit earlier than usual, but I had a lot to get done. After a quick breakfast of oatmeal and coffee, I headed to the lab. My current project involves a lot of data analysis, and I'm trying to write a script that automates part of the process. It’s a bit challenging, but I’m learning a lot, especially about machine learning.

    By noon, I was ready for a break. I met up with some friends from my program for lunch. It’s always refreshing to step away from work and just chill for a bit. We talked about a new AI paper that was recently published and speculated about its implications for our research.

    In the afternoon, I attended a seminar hosted by our department. The guest speaker discussed recent advances in quantum computing, and it was fascinating to see how far the technology has come. After the seminar, I spent a couple more hours at the lab, debugging a piece of code I’ve been stuck on for a few days.

    I wrapped up around 7 PM and headed home. Dinner was just some pasta and veggies—nothing fancy, as cooking isn’t exactly my forte. I spent the evening catching up on some required reading and preparing for tomorrow's group meeting. Before bed, I managed to squeeze in an episode of my current favorite TV show. It’s my little way to unwind after a busy day.
    """,
    """
    This morning started off with the usual hustle—grabbing a quick cup of coffee on my way to the office. I had a mountain of emails to tackle, but first, there was the weekly team meeting. We discussed our progress on various projects and set goals for the next week. My boss seemed pleased with the progress we're making, which is always a good sign.

    Lunch was at a new café near the office. I tried their quinoa salad, and it was surprisingly good. I might make it my new regular spot. In the afternoon, I had a long conference call with a client who's particularly detail-oriented. It took a while, but we managed to hash out the details of the project's next phase.

    After work, I stopped by the gym for a quick workout. It’s a good stress reliever, and it helps me transition from work mode to home mode. Once home, I cooked dinner, which was just some stir-fry chicken and veggies. Simple, but effective.

    The evening was pretty relaxed. I spent a couple of hours working on a puzzle. It's a 1000-piece one, and it's a picture of a beautiful seascape. Puzzles are my go-to relaxation activity. I chatted with a friend on the phone for a while before heading to bed around 11 PM. It was a pretty good day, all things considered.
    """,
    """
    Today was one of those days where I was reminded of the perks of freelancing. I started my morning by taking a walk in the park. It’s a luxury to start the day with some fresh air and not rush to an office. When I got back home, I settled into my workspace and checked my schedule. I had two client calls in the morning, both went well, and it looks like I’ll be starting on a new project soon.

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
