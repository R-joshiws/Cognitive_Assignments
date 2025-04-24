import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
import numpy as np
import matplotlib.pyplot as plt

# Q1: Text Processing
paragraph = """The cocaine trade significantly impacts economies, particularly in producing countries like Colombia, Peru, and Bolivia, where it generates substantial revenue and employment in cultivation, processing, and trafficking. It contributes to local economies by providing income for farmers and workers, though often in unstable and exploitative conditions. However, the illegal nature of the trade fuels corruption, violence, and instability, undermining formal economic systems and governance. Globally, the cocaine market, valued at billions annually, drives illicit financial flows, affecting banking systems and money laundering schemes. Efforts to combat the trade, such as crop eradication and law enforcement, divert significant public resources, straining national budgets. Despite its economic contributions, the social and political costs often outweigh short-term gains."""

# 1. Convert to lowercase and remove punctuation using re
text_lower = paragraph.lower()
text_no_punct = re.sub(r'[^\w\s]', '', text_lower)
print("Text without punctuation:", text_no_punct)

# 2. Tokenize into words and sentences
sentences = sent_tokenize(text_lower)
words_nltk = word_tokenize(text_no_punct)
print("Sentences:", sentences)
print("Words (NLTK):", words_nltk)

# 3. Split using split() and compare with word_tokenize()
words_split = text_no_punct.split()
print("Words (split):", words_split)
print("Difference: split() just splits on whitespace, but word_tokenize() handles tokens better, like splitting 'billions' and 'annually' properly.")

# 4. Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words_nltk if word not in stop_words]
print("Words after removing stopwords:", filtered_words)

# 5. Word frequency distribution
freq = {}
for word in filtered_words:
    freq[word] = freq.get(word, 0) + 1
print("Word frequency:", freq)

# Q2: Feature Extraction
# 1. Extract words with only alphabets
alpha_words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
print("\nWords with only alphabets:", alpha_words)

# 2. Remove stopwords
filtered_alpha_words = [word for word in alpha_words if word not in stop_words]
print("After removing stopwords:", filtered_alpha_words)

# 3. Stemming with PorterStemmer
porter = PorterStemmer()
stemmed_words = [porter.stem(word) for word in filtered_alpha_words]
print("Stemmed words:", stemmed_words)

# 4. Lemmatization with WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_alpha_words]
print("Lemmatized words:", lemmatized_words)

# 5. Compare stemmed and lemmatized outputs
print("Comparison: Stemming cuts words like 'economies' to 'economi', but lemmatization keeps 'economy'. I’d use lemmatization if I need real words for meaning, and stemming if I just need to reduce words for matching.")

# Q3: TF-IDF Analysis
texts = [
    "Cocaine trade boosts economies in Colombia with high revenue.",
    "Cocaine causes violence and corruption in local communities.",
    "Global cocaine market affects banking with illicit financial flows."
]

# 1. Bag of Words
count_vectorizer = CountVectorizer(lowercase=True, stop_words='english')
count_matrix = count_vectorizer.fit_transform(texts)
print("\n=== Bag of Words Representation ===")
print("Feature names:", count_vectorizer.get_feature_names_out())
print("Count Matrix:\n", count_matrix.toarray())

# 2. TF-IDF Scores
tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
print("\n=== TF-IDF Scores ===")
print("Feature names:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())

# 3. Top 3 keywords
feature_names = tfidf_vectorizer.get_feature_names_out()
for i, text in enumerate(texts):
    tfidf_scores = tfidf_matrix[i].toarray()[0]
    keyword_scores = zip(feature_names, tfidf_scores)
    top_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)[:3]
    print(f"Text {i+1}: {text}")
    print("Top 3 keywords:", [kw[0] for kw in top_keywords])
    if i == 0:
        print("Interpretation: 'colombia', 'revenue', 'economies' show economic impact.")
    elif i == 1:
        print("Interpretation: 'violence', 'corruption', 'communities' show social issues.")
    else:
        print("Interpretation: 'market', 'financial', 'banking' focus on global finance.")

# Q4: Similarity Analysis
tech_texts = [
    "AI improves automation with machine learning and neural networks.\nIt processes big data fast for better decisions.\nApplications include chatbots and image recognition.\nAI is changing industries like healthcare and finance.",
    "Blockchain ensures secure transactions with decentralized ledgers.\nIt uses cryptography to protect data integrity.\nPopular in cryptocurrencies like Bitcoin and Ethereum.\nBlockchain also supports smart contracts and supply chain."
]

# 1. Preprocess and tokenize
processed_texts = []
for text in tech_texts:
    text_lower = text.lower()
    text_clean = re.sub(r'[^\w\s]', '', text_lower)
    words = word_tokenize(text_clean)
    words = [w for w in words if w not in stop_words]
    processed_texts.append(words)

# 2a. Jaccard Similarity using sets
set1, set2 = set(processed_texts[0]), set(processed_texts[1])
jaccard_sim = len(set1.intersection(set2)) / len(set1.union(set2))
print("\nJaccard Similarity:", jaccard_sim)

# 2b. Cosine Similarity using TfidfVectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(processed_texts[0]), ' '.join(processed_texts[1])])
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
print("Cosine Similarity:", cosine_sim)

# 2c. Analyze
print("Analysis: Jaccard is low because the texts have few exact words in common. Cosine is better here since it captures similarity in meaning through TF-IDF, even with different words.")

# Q5: Sentiment Analysis
review = "This cocaine documentary is eye-opening and well-made.\nIt explains the trade’s impact clearly.\nI learned a lot about its effects.\nHighly recommend watching it!"

# 1. Polarity and subjectivity with TextBlob and VADER
blob = TextBlob(review)
vader = SentimentIntensityAnalyzer()
blob_scores = blob.sentiment
vader_scores = vader.polarity_scores(review)
print("\nTextBlob Polarity:", blob_scores.polarity, "Subjectivity:", blob_scores.subjectivity)
print("VADER Scores:", vader_scores)

# 2. Classify review
if vader_scores['compound'] > 0.05:
    sentiment = "Positive"
elif vader_scores['compound'] < -0.05:
    sentiment = "Negative"
else:
    sentiment = "Neutral"
print("Sentiment:", sentiment)

# 3. Word cloud for positive reviews
if sentiment == "Positive":
    wordcloud = WordCloud(width=400, height=200).generate(review)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('wordcloud.png')
    print("Word cloud saved as 'wordcloud.png'")

# Q6: Text Generation with LSTM
training_text = paragraph[:100]  # First 100 words
print("\nTraining Text:", training_text)

# 1. Tokenize with Keras Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([training_text])
total_words = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index

# Create input sequences
input_sequences = []
for line in training_text.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = np.array([np.eye(total_words)[i] for i in y])

# 2. Build LSTM model
model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
model.add(LSTM(50))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=50, verbose=0)

# 3. Generate text
seed_text = "cocaine"
next_words = 5
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in word_index.items():
        if index == predicted_word_index:
            seed_text += " " + word
            break
print("Generated Text:", seed_text)
