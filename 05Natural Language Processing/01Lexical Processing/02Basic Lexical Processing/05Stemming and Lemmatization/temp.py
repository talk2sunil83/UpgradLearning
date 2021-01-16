# %%
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
# %%
%time

word = "schooling"

# instantiate wordnet lemmatizer
lemmatizer = WordNetLemmatizer()  # write code here

# lemmatize word
lemmatized = lemmatizer.lemmatize(word, pos='v')  # write code here. Pass the parameter -> pos='v' to the lemmatize function to lemmatize verbs correctly.

# print lemmatized word -- don't change the following code, it is used to evaluate your code
print(lemmatized)
# %%
