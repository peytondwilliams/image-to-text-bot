import praw
import pdb
import re
import os
import requests

from text_recognition import main

reddit = praw.Reddit('bot1')

if not os.path.isfile("posts_replied_to.txt"):
    posts_replied_to = []

else:
    with open("posts_replied_to.txt", "r") as f:
        posts_replied_to = f.read()
        posts_replied_to = posts_replied_to.split("\n")
        posts_replied_to = list(filter(None, posts_replied_to))


subreddit = reddit.subreddit("imgtotextbotsandbox")

for submission in subreddit.hot(limit=10):
    if submission.id not in posts_replied_to:
        if submission.is_reddit_media_domain and submission.domain == 'i.redd.it':
            url = submission.url
            file_name = submission.id + ".jpg"
            imgre = requests.get(url)

            with open(file_name, "wb") as f:
                f.write(imgre.content)

            #put text recognition call here
            read_text = main(file_name)
            os.remove(file_name)

            submission.reply(read_text)
            posts_replied_to.append(submission.id)


with open("posts_replied_to.txt", "w") as f:
    for post_id in posts_replied_to:
        f.write(post_id + "\n")           



"""
for submission in subreddit.hot(limit=5):
    print("Title: ", submission.title)
    print("Text: ", submission.selftext)
    print("Score: ", submission.score)
    print("----------------------------------\n")
"""