#!/usr/bin/env python
# coding: utf-8

# ## To Lunch Spark

# In[3]:


sc


# In[4]:


sc.master


# ## Configure Other Package with Spark

# In[5]:


# !pip install findspark
import findspark
findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession.builder     .appName("PySpark MySQL Connection")     .config("spark.jars", "/usr/share/java/mysql-connector-j-8.0.33.jar")     .config("spark.mongodb.input.uri", "mongodb://localhost:27017/health.tweetd")     .config("spark.mongodb.output.uri", "mongodb://localhost:27017/health.tweetd")     .getOrCreate()


# In[12]:


# Define Hadoop input path
hadoop_input_path = "hdfs://localhost:9000/health/tweets.csv"

# Load data from Hadoop as a DataFrame
data_df = spark.read.format("csv").option("header", "true").load(hadoop_input_path)
# Display the loaded data
data_df.show(2)


# ## Visualise the Schema

# In[13]:


# Display the loaded data
data_df.printSchema()


# In[14]:


# data_df.write \
#   .format("jdbc") \
#   .option("driver","com.mysql.jdbc.Driver") \
#   .option("url", "jdbc:mysql://localhost:3306/health") \
#   .option("dbtable", "tweetd") \
#   .option("user", "root") \
#   .option("password", "password") \
#   .save()


# ## Load to mysql

# In[81]:


mysql_df = spark.read   .format("jdbc")   .option("driver","com.mysql.jdbc.Driver")   .option("url", "jdbc:mysql://localhost:3306/health")   .option("dbtable", "tweetd")   .option("user", "root")   .option("password", "password")   .load()


# ## Load to mysqlMongodb

# In[16]:


# Install the pyspark-mongo connector
# !pip install pyspark pymongo


# In[17]:


# Write data to MongoDB
# mongo_df = mysql_df.write.format("mongo").save()


# In[18]:


# Read data from MongoDB collection
mongo_read = spark.read.format("mongo").load()


# In[19]:


# Display the loaded data
mongo_read.printSchema()


# In[20]:


# Show the data
mongo_read.show(1)


# ## Load Library for Sentiment

# In[21]:


from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import DataFrameWriter
from textblob import TextBlob
from nltk.corpus import stopwords 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer() 
eng_stop_word = list(stopwords.words('english'))


# ## Data Cleaning

# In[22]:


def ProcessedTweets(text):
    #changing tweet text to small letter
    text = text.lower()
    #removing @ and links
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w+:\/\/\S+)", " ", text).split())
    #removing repeating characters
    text = re.sub(r'\@\w+|\#\w+|\d+', '', text)
    #removing puntuations and number
    punct = str.maketrans('','',string.punctuation+string.digits)
    text = text.translate(punct)
    #tokenising words and removing stop words from the tweet text
    tokens = word_tokenize(text)
    eng_stop_word = list(stopwords.words('english'))
    filtered_words = [w for w in tokens if w not in eng_stop_word]
#     filtered_words = [w for w in filtered_words if w not in emoji]
    #lemmetizingwords
    lemmatizer =WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    text = " ".join(lemma_words)
    return text   

#function for polarity score
def polarity(tweet):
    return TextBlob(tweet).sentiment.polarity


#getting sentiment type and setting condition
def sentimenttextblob(polarity):
    if polarity < 0:
        return "Negative"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Positive"


# In[23]:


# df = mongo_read.toPandas()
# df.shape


# In[38]:


def get_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    sentiment = sentiment_scores['compound']
    
    if sentiment >= 0.05:
        return 'Positive'
    elif sentiment <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


# In[39]:


sentiment_udf = udf(get_sentiment, StringType())


# In[40]:


# Clean the tweet data by selecting relevant columns and removing unwanted characters
cleaned_data = mongo_read.select("Date","ID", "tweet")     .na.drop(subset=["tweet"])     .withColumn("tweet", lower(regexp_replace("tweet", "[^a-zA-Z\\s]", "")))

# Apply sentiment analysis using the UDF
sentiment_data = cleaned_data.withColumn("sentiment", sentiment_udf(cleaned_data["tweet"]))


# In[ ]:





# In[41]:


sentiment_data.select('Date').show(1)


# ## Analyze distribution 

# In[42]:


# Analyze the sentiment distribution
sentiment_distribution = sentiment_data.groupBy("sentiment").count()
# Show the final result
sentiment_data.show()


# ## Clean and transform tweet column for Models

# In[80]:


# Clean and transform tweet column
def clean_and_transform(tweet):
    cleaned_tweet = regexp_replace(tweet, r"[^a-zA-Z0-9\s]", "")
    lowercase_tweet = lower(cleaned_tweet)
    words = lowercase_tweet.split(" ")
    return words

clean_transform_udf = udf(clean_and_transform, ArrayType(StringType()))

df_transformed = sentiment_data.withColumn("transformed_tweets", clean_transform_udf(sentiment_data["tweet"]))

# Vectorize the transformed tweets
tokenizer = Tokenizer(inputCol="transformed_tweets", outputCol="words")
vectorizer = CountVectorizer(inputCol="words", outputCol="features")

# Create a logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="sentiment")

# Create a pipeline with vectorization and the logistic regression model
pipeline = Pipeline(stages=[tokenizer, vectorizer, lr])

# Split the data into training and test sets
train_data, test_data = df_transformed.randomSplit([0.8, 0.2], seed=42)

# Train the model
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)


# In[77]:


sentiment_data.select('tweet').show(2)


# In[67]:


remover


# In[76]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, Tokenizer
from pyspark.ml.classification import LogisticRegressionModel


# In[ ]:





# In[ ]:





# In[ ]:





# In[75]:


# # Feature extraction
# stopwords = StopWordsRemover.loadDefaultStopWords("english")
# remover = StopWordsRemover(inputCol="tweet", outputCol="filtered_tweet", stopWords=stopwords)
# data = sentiment_data.filter(col("tweet").isNotNull() & (col("tweet").getItem(0) != ""))
# filtered_data = remover.transform(data)

# vectorizer = CountVectorizer(inputCol="tweet", outputCol="features")
# vectorizer_model = vectorizer.fit(filtered_data)
# featured_data = vectorizer_model.transform(filtered_data)


# In[47]:


# Split the data into training and testing sets
(training_data, testing_data) = featured_data.randomSplit([0.8, 0.2], seed=42)

# Train the sentiment analysis model using logistic regression
lr = LogisticRegression(labelCol="sentiment", featuresCol="features")
lr_model = lr.fit(training_data)


# In[48]:



# Make predictions on the testing data
predictions = lr_model.transform(testing_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="sentiment")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Forecast

# In[24]:


import datetime
from pyspark.sql.types import TimestampType
# Forecast sentiment for different time intervals
# Forecast sentiment for different time intervals
date_format = "yyyy-MM-dd HH:mm:ss"
end_date = datetime.datetime.strptime("2023-05-25 00:00:00", date_format)


# In[25]:


# 1 week forecast
start_date_1w = end_date - datetime.timedelta(weeks=1)
forecast_1w = sentiment_data.filter((col("Date") > start_date_1w) & (col("Date") <= end_date))

# 1 month forecast
start_date_1m = end_date - datetime.timedelta(days=30)
forecast_1m = sentiment_data.filter((col("Date") > start_date_1m) & (col("Date") <= end_date))

# 3 month forecast
start_date_3m = end_date - datetime.timedelta(days=90)
forecast_3m = sentiment_data.filter((col("Date") > start_date_3m) & (col("Date") <= end_date))

# Aggregate sentiment counts for each forecast period
sentiment_counts_1w = forecast_1w.groupBy("sentiment").count()
sentiment_counts_1m = forecast_1m.groupBy("sentiment").count()
sentiment_counts_3m = forecast_3m.groupBy("sentiment").count()


# In[ ]:


# Show the forecasted sentiment counts
print("1 Week Forecast:")
sentiment_counts_1w.show()

print("1 Month Forecast:")
sentiment_counts_1m.show()

print("3 Month Forecast:")
sentiment_counts_3m.show()# 1 week forecast


# In[ ]:


start_date_1w = end_date - datetime.timedelta(weeks=1)
forecast_1w = sentiment_data.filter((col("Date") > start_date_1w) & (col("Date") <= end_date))

# 1 month forecast
start_date_1m = end_date - datetime.timedelta(days=30)
forecast_1m = sentiment_data.filter((col("Date") > start_date_1m) & (col("Date") <= end_date))

# 3 month forecast
start_date_3m = end_date - datetime.timedelta(days=90)
forecast_3m = sentiment_data.filter((col("Date") > start_date_3m) & (col("Date") <= end_date))


# In[ ]:



# Aggregate sentiment counts for each forecast period
sentiment_counts_1w = forecast_1w.groupBy("sentiment").count()
sentiment_counts_1m = forecast_1m.groupBy("sentiment").count()
sentiment_counts_3m = forecast_3m.groupBy("sentiment").count()


# In[ ]:


# Show the forecasted sentiment counts
print("1 Week Forecast:")
sentiment_counts_1w.show()

print("1 Month Forecast:")
sentiment_counts_1m.show()

print("3 Month Forecast:")
sentiment_counts_3m.show()


# In[31]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[34]:


plt.pie(sentiment_data, labels = ['Positive', 'Neutral', 'Negative'], autopct = '%1.2f%%')
circle = plt.Circle( (0,0), 0.4, color='white')
p = plt.gcf()
p.gca().add_artist(circle)
plt.show()


# In[37]:


nltk.download('omw-1.4')

# convert the tweet text into a string separate with " "
sentiment_data['Processed_Tweets'] = sentiment_data['Text'].apply(ProcessedTweets)
tweets_string = sentiment_data['Processed_Tweets'].tolist()
tweets_string = " ".join(tweets_string)


# In[ ]:


# Displaying the most talked about word in a word cloud 
# some stop words were still evident 
# Instantiate the Twitter word cloud object
w_cloud = WordCloud(collocations = False,max_words=200, background_color = 'white', width = 7000, height = 5000).generate(tweets_string)

# Display the generated Word Cloud
plt.imshow(w_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:





# In[ ]:




