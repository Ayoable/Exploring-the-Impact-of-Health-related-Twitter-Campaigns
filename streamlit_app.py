import streamlit as st
import pandas as pd
from textblob import TextBlob

# Streamlit app
st.title("Tweet Sentiment Analysis")

# Load tweet data from CSV
data = pd.read_csv("training.1600000.processed.noemoticon.csv', names=["target", "ids", "date", "flag", "user", "text"], encoding="ISO-8859-1", chunksize= 100)

# Display the loaded tweet data
st.subheader("Loaded Tweet Data")
st.dataframe(data)
# Perform sentiment analysis on the loaded tweet data
sentiment_scores = [TextBlob(tweet).sentiment.polarity for tweet in data['text']]
    

# Calculate overall sentiment score
overall_score = sum(sentiment_scores) / len(sentiment_scores)
st.write
# Classify overall sentiment
if overall_score > 0:
    sentiment = "Positive"
elif overall_score < 0:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

# Display results
st.success(f"Overall Sentiment: {sentiment}")
st.bar_chart(sentiment_scores)

# Filter data for one week
one_week_data = data[data['date'] >= pd.to_datetime('today') - pd.DateOffset(weeks=1)]
one_week_sentiment = []
for tweet in one_week_data['text']:
    analysis = TextBlob(tweet)
    one_week_sentiment.append(analysis.sentiment.polarity)
one_week_average_sentiment = sum(one_week_sentiment) / len(one_week_sentiment)

# Filter data for one month
one_month_data = data[data['date'] >= pd.to_datetime('today') - pd.DateOffset(months=1)]
one_month_sentiment = []
for tweet in one_month_data['text']:
    analysis = TextBlob(tweet)
    one_month_sentiment.append(analysis.sentiment.polarity)
one_month_average_sentiment = sum(one_month_sentiment) / len(one_month_sentiment)

# Filter data for three months
three_months_data = data[data['date'] >= pd.to_datetime('today') - pd.DateOffset(months=3)]
three_months_sentiment = []
for tweet in three_months_data['text']:
    analysis = TextBlob(tweet)
    three_months_sentiment.append(analysis.sentiment.polarity)
three_months_average_sentiment = sum(three_months_sentiment) / len(three_months_sentiment)

# Display sentiment forecasts
st.header("Sentiment Forecasts")
st.subheader("One Week")
st.write(f"Average Sentiment: {one_week_average_sentiment}")
st.bar_chart(one_week_sentiment)

st.subheader("One Month")
st.write(f"Average Sentiment: {one_month_average_sentiment}")
st.bar_chart(one_month_sentiment)

st.subheader("Three Months")
st.write(f"Average Sentiment: {three_months_average_sentiment}")
st.bar_chart(three_months_sentiment)
