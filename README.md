# image-to-text-bot
A reddit bot which uses OCR to find and transcribe text in images. Text is then left as a comment on a reddit post.

### Why?
The idea came to me as I was brainstorming interesting projects I could work on which involve computer vision. I believe this project has the potential to solve multiple problems.

- __Search Engine Optimization__. An image post cannot be found from the text within the image. Transcribing the text allows for easier look up by the text content of the image.  
- __Usability for the visually impared__. Many reddit users are not able to enjoy memes and other images posted on the platform due to their disability. This bot will allow them to better enjoy this type of content with the addition of text-to-speech or similar tools.
- __Data Usage Saving__. An individual may want to save on their date usage due to limited bandwitdh or data cap. They can use the bot to read the text from the image without having to load the image themselves. This is especially relevant as images of tweets and other solid blobs of text are becoming increasingly common.

### Main Technologies Used:
- Python  
- OpenCV (Text Detection)  
- Tesseract (Text Recognition)  
- PRAW (Reddit API Client)  
- AWS EC2 (Server Hosting)

### Check out the bot's past performance on reddit:
[u/imgtotextbot](https://www.reddit.com/user/imgtotextbot)

### Or click one of the links here:
https://www.reddit.com/r/imgtotextbotsandbox/comments/ecwvr4/geeks/
https://www.reddit.com/r/imgtotextbotsandbox/comments/ebpz3m/example_02/
https://www.reddit.com/r/imgtotextbotsandbox/comments/ecuz2e/text_test/
