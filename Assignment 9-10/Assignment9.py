# Lab Assignment 9: NLP using Python
# I'm using the paragraph about cocaine economy as my input
# I'll do all questions (Q1-Q4) in one script

import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK data (I did this separately: nltk.download('punkt'), nltk.download('stopwords'), nltk.download('wordnet'))
# The paragraph
text = """The cocaine trade significantly impacts economies, particularly in producing countries like Colombia, Peru, and Bolivia, where it generates substantial revenue and employment in cultivation, processing, and trafficking. It contributes to local economies by providing income for farmers and workers, though often in unstable and exploitative conditions. However, the illegal nature of the trade fuels corruption, violence, and instability, undermining formal economic systems and governance. Globally, the cocaine market, valued at billions annually, drives illicit financial flows, affecting banking systems and money laundering schemes. Efforts to combat the trade, such as crop eradication and law enforcement, divert significant public resources, straining national budgets. Despite's economic contributions, the social and political costs often outweigh short-term gains."""

# Q1: Text Processing
print("=== Q1: Text Processing ===")

# 1. Convert to lowercase and remove punctuation
text_lower = text.lower()
text_no_punct = text_lower.translate(str.maketrans("", "", string.punctuation))
print("Text without punctuation:", text_no_punct)

# 2. Tokenize into words and sentences
words = word_tokenize(text_no_punct)
sentences = sent_tokenize(text_lower)
print("Sentences:", sentences)
print("Words:", words)

# 3. Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]
print("Words after removing stopwords:", filtered_words)

# 4. Word frequency distribution
freq = {}
for word in filtered_words:
    if word in freq:
        freq[word] += 1
    else:
        freq[word] = 1
print("Word frequency:", freq)

# Q2: Stemming and Lemmatization
print("\n=== Q2: Stemming and Lemmatization ===")

# 1. Use filtered words from Q1
porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

porter_stems = [porter.stem(word) for word in filtered_words]
lancaster_stems = [lancaster.stem(word) for word in filtered_words]
lemmas = [lemmatizer.lemmatize(word) for word in filtered_words]

# Compare results
print("Porter Stemming:", porter_stems)
print("Lancaster Stemming:", lancaster_stems)
print("Lemmatization:", lemmas)

# Q3: Regular Expressions and Text Splitting
print("\n=== Q3: Regular Expressions and Text Splitting ===")

# 1. Regex tasks
# a. Words with more than 5 letters
long_words = re.findall(r'\b\w{6,}\b', text_lower)
print("Words with more than 5 letters:", long_words)

# b. Extract numbers (none in this text, but I'll try anyway)
numbers = re.findall(r'\d+\.?\d*', text)
print("Numbers:", numbers)

# c. Capitalized words
capitalized = re.findall(r'\b[A-Z][a-z]*\b', text)
print("Capitalized words:", capitalized)

# 2. Text splitting
# a. Words with only alphabets
alpha_words = re.findall(r'\b[a-zA-Z]+\b', text)
print("Words with only alphabets:", alpha_words)

# b. Words starting with a vowel
vowel_words = re.findall(r'\b[aeiouAEIOU][a-zA-Z]*\b', text)
print("Words starting with a vowel:", vowel_words)

# Q4: Custom Tokenization & Regex-based Text Cleaning
print("\n=== Q4: Custom Tokenization & Regex-based Text Cleaning ===")

# 1. Custom tokenization function
def custom_tokenize(text):
    # Keep contractions and hyphenated words, tokenize numbers separately
    # First, separate numbers (keep decimals)
    text = re.sub(r'(\d+\.\d+)', r' \1 ', text)
    # Tokenize while keeping contractions and hyphens
    tokens = []
    current_token = ""
    i = 0
    while i < len(text):
        if text[i].isalnum() or text[i] == "'":
            current_token += text[i]
        elif text[i] == '-' and current_token and i + 1 < len(text) and text[i + 1].isalnum():
            current_token += text[i]
        else:
            if current_token:
                tokens.append(current_token)
                current_token = ""
            if text[i] not in string.punctuation or text[i] == '-':
                tokens.append(text[i])
        i += 1
    if current_token:
        tokens.append(current_token)
    return [t for t in tokens if t and t not in string.punctuation]

tokens = custom_tokenize(text)
print("Custom tokens:", tokens)

# 2. Regex substitutions
# a. Replace emails
text_cleaned = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '<EMAIL>', text)
# b. Replace URLs
text_cleaned = re.sub(r'https?://[^\s]+', '<URL>', text_cleaned)
# c. Replace phone numbers
text_cleaned = re.sub(r'\b(\+\d{10,12}|\d{3}-\d{3}-\d{4})\b', '<PHONE>', text_cleaned)
print("Text after cleaning:", text_cleaned)
