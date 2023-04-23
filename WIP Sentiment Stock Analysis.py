#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

positive_texts = ['bullish', 'upswing',  'growth',  'profitable', 'high yield', 'buy', 'positive earnings',  'strong',   'beating estimates',    'outperform',    'undervalued',    'dividends',    'cash flow',    'earnings beat',    'positive guidance',    'price increase',    'good news',    'market gain',    'inflation protection',    'upward trend',    'momentum',    'buyout',    'merger',    'acquisition',    'success',    'outstanding',    'advantage',    'breakout',    'growth stock',    'dividend payout',    'bull market',    'record high',    'strong earnings',    'rising stock price', 'expansion',  'innovative',  'value stock',  'prosperous',  'price surge',  'blue chip', 'outstanding performance',  'upward momentum',  'positive outlook',  'cash cow',  'buy recommendation',  'solid fundamentals', 'profit margin', 'positive surprises',  'bullish sentiment', 'love', 'english', 'tv', 'good', 'particular', 'investigative', 'detective', 'stuff', 'this', 'guy', 'is', 'really', 'appeal', 'real', 'beats', 'comedy', 'pretty', 'much', 'find', 'think', 'like', "This restaurant is amazing!", "The concert was fantastic!", "admire", "affectionate", "amazing", "amusing", "appreciative", "awesome", "beautiful", "brilliant", "captivating", "carefree", "cheerful", "comfortable", "compassionate", "confident", "courageous", "creative", "dazzling", "delightful", "desirable", "eager", "ecstatic", "efficient", "electrifying", "energetic", "enjoyable", "enthusiastic", "excellent", "exciting", "exquisite", "fabulous", "fantastic", "fascinating", "friendly", "fun", "generous", "genuine", "glorious", "good-looking", "graceful", "grateful", "great", "happy", "harmonious", "helpful", "hilarious", "hopeful", "humorous", "imaginative", "impressive", "incredible", "ingenious", "inspiring", "interesting", "intriguing", "intuitive", "joyful", "kind", "knowledgeable", "likable", "lively", "lovely", "lucky", "magnificent", "marvelous", "mesmerizing", "mind-blowing", "motivating", "natural", "neat", "nice", "optimistic", "outstanding", "passionate", "peaceful", "perfect", "playful", "pleasant", "pleasing", "positive", "powerful", "precious", "productive", "prosperous", "proud", "radiant", "refreshing", "relaxing", "remarkable", "resilient", "resourceful", "rich", "romantic", "satisfying", "sensational", "serene", "skillful", "smiling", "spectacular", "splendid", "stimulating"]
negative_texts = [ 'sell-off', 'downtrend', 'crash', 'collapse', 'slump', 'plunge', 'decline', 'selloff', 'sellout', 'downward', 'losses', 'volatile', 'overbought', 'oversold', 'negative', 'uncertainty', 'bear', 'bears', 'correction', 'debt', 'default', 'deflation', 'devaluation', 'discouraging', 'disappointing', 'downturn', 'drawdown', 'economic crisis', 'economic slowdown', 'inflation', 'low', 'negative growth', 'recession', 'reduced', 'sell', 'sell signal', 'short', 'short position', 'slow', 'slowdown', 'sluggish', 'stagflation', 'trend reversal', 'underperformance', 'underperforming', 'volatility', 'weak', 'weaken', 'weakness', 'expectations', 'boring', 'awful', 'no', 'neither', 'blank', 'seem', 'skip', 'bad', "overtaking Tesla", "aggravating", "anxiety", "arrogant", "ashamed", "awkward", "bitter", "boring", "bruised", "brutal", "burdensome", "clumsy", "cold", "complaining", "confused", "cowardly", "cruel", "crushing", "damaging", "deadly", "deafening", "decaying", "defeated", "deficient", "deformed", "dejected", "delicate", "demanding", "depraved", "depressed", "deranged", "despairing", "destructive", "detestable", "dim", "dirty", "disappointing", "disgusting", "disheartening", "dishonest", "dismal", "displeasing", "disrespectful", "dissatisfied", "distasteful", "distressing", "dreadful", "dull", "embarrassing", "enraged", "evil", "excruciating", "exhausting", "expensive", "extravagant", "fading", "fake", "fearful", "feeble", "filthy", "foolish", "frustrating", "ghastly", "gloomy", "grim", "gross", "gruesome", "guilty", "hard", "harsh", "heartbreaking", "heavy", "hideous", "horrendous", "horrible", "hostile", "humiliating", "hurtful", "ignorant", "ill", "imperfect", "impolite", "impractical", "incompetent", "inconvenient", "indecisive", "ineffective", "inferior", "injurious", "insane", "insensitive", "insipid", "insulting", "intolerable", "irrelevant", "irritating", "joyless", "lame", "crashed", "lawsuit", "sued", "bankrupt", "default", "scandal", "fraud", "crisis", "debt", "downfall", "slump", "bearish", "sell-off", "correction", "downturn", "plunge", "decline", "losses", "crash", "recession", "fall", "bankruptcy", "failure", "tumble", "selloff", "meltdown", "liquidation", "negative", "poor", "weakness", "unstable", "bear", "sell", "dump", "devaluation", "drop", "plummet", "deficit", "class-action lawsuit", 'concerned', "face backlash", "backlash"]
# Create training data by combining positive and negative texts
text_training = positive_texts + negative_texts

# Create labels for the training data
text_labels = [1] * len(positive_texts) + [0] * len(negative_texts)

# Create a CountVectorizer object to transform the texts into a matrix of word counts
text_counter = CountVectorizer()

# Fit the CountVectorizer object to the training data
text_counter.fit(text_training)

# Transform the training data into a matrix of word counts
text_counts = text_counter.transform(text_training)

# Train a Multinomial Naive Bayes classifier on the text data
text_classifier = MultinomialNB()
text_classifier.fit(text_counts, text_labels)

# Save the trained classifier and vectorizer using pickle
with open("text_classifier.pkl", "wb") as f:
    pickle.dump(text_classifier, f)
    
with open("text_counter.pkl", "wb") as f:
    pickle.dump(text_counter, f)
    pickle.dump(text_counter, f)


# In[2]:


import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the saved classifier and vectorizer
with open("text_classifier.pkl", "rb") as f:
    text_classifier = pickle.load(f)

with open("text_counter.pkl", "rb") as f:
    text_counter = pickle.load(f)

# Define new text to classify
intercepted_text = "You smell aboslutely horrible"

# Transform the new text into a matrix of word counts
text_counts = text_counter.transform([intercepted_text])

# Use the trained classifier to predict the sentiment of the new text
final_pos = text_classifier.predict_proba(text_counts)[0][1]
final_neg = text_classifier.predict_proba(text_counts)[0][0]

# Print the result
if final_pos > final_neg:
    print("The text is positive.")
else:
    print("The text is negative.")


# In[3]:


import requests
import csv

# Set up the News API endpoint and parameters
url = "https://newsapi.org/v2/everything?q=tesla%20car&sortBy=publishedAt&language=en&pageSize=100&apiKey=9c96db10519d4d498dae63e142338b62"

# Send the request to the API and retrieve the results
response = requests.get(url)
results = response.json()["articles"]

# Sort the articles by published time, in descending order (most recent first)
results = sorted(results, key=lambda article: article["publishedAt"], reverse=True)

# Open a CSV file to write the results
with open("tesla_news.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Title", "Description", "URL", "Content", "PublishedAt"])

    # Write each relevant article to the CSV file
    for article in results:
        title = article["title"].lower()
        description = article["description"].lower()
        content = article["content"].lower()

        # Check if the article contains relevant keywords
        if "tesla" in title and "car" in (description + content):
            writer.writerow([article["title"], article["description"], article["url"], article["content"], article["publishedAt"]])


# In[4]:


import os

# List all files in the current directory
print(os.listdir())


# In[5]:


# Next is to install textblob then now I have to read the tesla_news.csv file.
import pandas as pd
from textblob import TextBlob

# Load the CSV file
df = pd.read_csv("tesla_news.csv", usecols=["PublishedAt", "Content"])

# Define a function to get the sentiment of a piece of text
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment

# Apply the sentiment function to the "Content" column of the DataFrame
df["Sentiment"] = df["Content"].apply(get_sentiment)



# In[6]:


import pickle
import csv
from sklearn.feature_extraction.text import CountVectorizer

# Load the saved classifier and vectorizer
with open("text_classifier.pkl", "rb") as f:
    text_classifier = pickle.load(f)

with open("text_counter.pkl", "rb") as f:
    text_counter = pickle.load(f)

# Open the CSV file of news articles
with open("tesla_news.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    # Process each article
    for row in reader:
        # Extract the relevant text from the article
        text = row["Title"] + " " + row["Description"] + " " + row["Content"]

        # Transform the text into a matrix of word counts
        text_counts = text_counter.transform([text])

        # Use the trained classifier to predict the sentiment of the text
        final_pos = text_classifier.predict_proba(text_counts)[0][1]
        final_neg = text_classifier.predict_proba(text_counts)[0][0]

        # Print the result
        if final_pos > final_neg:
            print("The article '{}' is positive.".format(row["Title"]))
        else:
            print("The article '{}' is negative.".format(row["Title"]))


# In[7]:


import pandas as pd
import plotly.graph_objs as go


# Create a line chart of the sentiment scores over time
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["PublishedAt"], y=df["Sentiment"], mode="lines", line=dict(color="green", width=3)))
fig.update_layout(title="Tesla News Sentiment Score", xaxis_title="Published Date", yaxis_title="Sentiment Score")

# Show the chart
fig.show()

# I'll also look into reinforcing/actively adding data through other apis or urls such as Nasdaq or YahooFinance
# Test more data of Pos and Neg
# Finally implementing a stock indicator and then combine it with this


# In[ ]:




