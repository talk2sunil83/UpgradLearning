# %%
import requests
from nltk import FreqDist
from nltk.corpus import stopwords

# load the ebook
url = "https://www.gutenberg.org/files/16/16-0.txt"
peter_pan = requests.get(url, verify=False).text

# break the book into different words using the split() method
peter_pan_words = peter_pan.split()

# build frequency distribution using NLTK's FreqDist() function
word_frequency = FreqDist(peter_pan_words)

# extract nltk stop word list
stopwords = stopwords.words('english')

# remove 'stopwords' from 'peter_pan_words'
no_stops = [w for w in peter_pan_words if w not in stopwords]  # write code here

# create word frequency of no_stops
word_frequency = FreqDist(no_stops)  # write code here


# extract the most frequent word and its frequency
frequency = word_frequency.most_common(1)[0][1]

# print the third most frequent word - don't change the following code, it is used to evaluate the code
print(frequency)
# %%
