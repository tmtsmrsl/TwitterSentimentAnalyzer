---
title: Twitter Sentiment Analysis - Introduction and Data Collection (Part 1)
post_excerpt: Emd-to-end machine learning project on sentiment analysis. In this post, we will walk through the data collection process with distant supervision method.
taxonomy:
    category:
        - knowledge
        - notes
---


# End-to-End Machine Learning Project: Twitter Sentiment Analysis - Introduction and Data Collection (Part 1)

### Introduction

I believe that creating a portfolio project is a great way to practice and showcase our skills in data science. In this post, I want to share an example of an end-to-end machine learning project on sentiment analysis, which is a rapidly growing field in natural language processing and machine learning. We will go over the entire process, from data collection and preprocessing to model building, creating a dashboard, and finally deploying the model and dashboard as an online application.

### Data Collection

Before collecting the data, we need to define the objective of our project. Our objective is to predict the public's sentiment about a brand (product, service, company or person) based on tweet data. We will use the data collection methodology described in [this paper](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf) (Twitter Sentiment Classification using Distant Supervision, Go, Bhayani, & Huang, 2009). 


Distant supervision is a method that utilizes a set of rules to automatically label a dataset. Since it does not require human intervention, it can save a lot of time and resources, especially when working with large datasets. In our case, we will use emoticons to label the sentiment of the tweet. Specifically, a tweet with a smiley face emoticon will be labeled as positive, and a tweet with a frowning face emoticon will be labeled negative. We will use a library called `snscrape` to collect the tweets; it does not require using the Twitter API, so we can retrieve a large amount of tweets without worrying about the [rate limit](https://developer.twitter.com/en/docs/twitter-api/rate-limits). In the following section we will walk through the codes and explain the logic behind them.

**Disclaimer: As of 11 January 2023, Twitter modified its frontend API and the code below will no longer work. I will provide an alternative as soon as I find a solution.**

First we will import the necessary libraries, please install them first if you do not already have them.


```python
!pip install snscrape
import snscrape.modules.twitter as sntwitter
import datetime as dt
import pandas as pd
```

Then we will make a function that utilizes `sntwitter.TwitterSearchScraper` to retrieve the tweets and save them in a dataframe. The function takes the following arguments:  
* search_term: the term you want to search for on Twitter
* start_date: the start date of the search range in the format of datetime.date object
* end_date: the end date of the search range in the format of datetime.date object
* num_tweets: the number of tweets you want to retrieve

A `for` loop is used to iterate over and store the tweet data (username, date, and tweet content) returned by the `get_items` method of `sntwitter.TwitterSearchScraper`. We use lang:en (English language) and exclude:retweets as the search filters. The tweet data is finally returned as a dataframe.


```python
def scrape_tweet(search_term, start_date, end_date, num_tweets):
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    tweet_data = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper('{} since:{} until:{} lang:en exclude:retweets'.format(search_term, start_date, end_date)).get_items()):
        if i >= num_tweets:
            break
        tweet_data.append([tweet.user.username, tweet.date, tweet.content])
    tweet_df = pd.DataFrame(tweet_data, columns=['username', 'date', 'tweet'])
    return tweet_df
```

For this project, we want to retrieve tweets from 2022-01-01 to 2022-12-31. So we make another function, `daily_scrape_2022` which utilizes the `scrape_tweet` function to retrieve tweets for each day in 2022. We can specify the number of tweets we want to retrieve for each day using `num_daily`.


```python
def daily_scrape_2022(search_term, num_daily):
    start_date = dt.datetime(2022, 1, 1)
    end_date = dt.datetime(2022, 1, 2)
    delta = dt.timedelta(days=1)
    df = pd.DataFrame()
    for n in range(365):
        temp_df = scrape_tweet(search_term, start_date, end_date, num_daily)
        df = pd.concat([df, temp_df])
        start_date += delta
        end_date += delta
    return df
```

Now we will use the `daily_scrape_2022` function to retrieve 1000 tweets daily for each day in 2022. Tweets with negative sentiment will be searched with the term ":(" while tweets with positive sentiment will be searched with the term ":)".


```python
ori_neg_df = daily_scrape_2022(":(", 1000)
ori_pos_df = daily_scrape_2022(":)", 1000)
```

The retrieved tweets do not always contain the specified search term, so we need to do some filtering. We create two functions, `filter_include` to include tweets containing a specific term and `filter_exclude` to exclude tweets containing a specific term. Note that both functions take a list of terms as the second argument, so we can filter multiple terms at once. 


```python
def filter_include(df, term_list):
    temp_df = pd.DataFrame()
    for term in term_list:
        add_df = df[df['tweet'].str.contains(term, regex=False) == True]
        temp_df = pd.concat([temp_df, add_df]).drop_duplicates(ignore_index=True)
    return 

def filter_exclude(df, term_list):
    temp_df = df.copy()
    for term in term_list:
        temp_df = temp_df[temp_df['tweet'].str.contains(term, regex=False) == False]
    return temp_df
```

For the negative tweets, first we will include tweets containing the term ":(" or ":-(", then exclude tweets containing the term ":)", ":D", or ":-)". Note that `filter_exclude` is done on `neg_df`, not `ori_neg_df`. Tweets with smiley face emoticon are excluded because we do not want to label tweets containing both frowning face and smiley face emoticons as negative. After filtering, we have 358624 tweets with negative sentiment.


```python
neg_df = filter_include(ori_neg_df, [":(", ":-("])
neg_df = filter_exclude(neg_df, [":)", ":D", ":-)"])
neg_df.shape
```




    (358624, 3)



Similar filtering is done for the positive tweets. After filtering, we have 343477 tweets with positive sentiment.


```python
pos_df = filter_include(ori_pos_df, [":)", ":D", ":-)"])
pos_df = filter_exclude(pos_df, [":(", ":-("])
pos_df.shape
```




    (343477, 3)



Next, we will remove all the emoticons used for `filter_include` from the tweets. We do this because we want our model to classify the sentiment based on the words instead of emoticons. If we include the emoticons in the training data, the model will have poor generalization performance because real-world data may not contain emoticons.


```python
def remove_term(df, term_list):
    temp_df = df.copy()
    for term in term_list:
        temp_df['tweet'] = temp_df['tweet'].str.replace(term, " ", regex=False)
    return temp_df
```


```python
neg_df = remove_term(neg_df, [":(", ":-("])
pos_df = remove_term(pos_df, [":)", ":D", ":-)"])
```

Last we will label the sentiment of the tweets and combine them into one dataframe.


```python
neg_df["sentiment"] = "Negative"
pos_df["sentiment"] = "Positive"
df = pd.concat([neg_df, pos_df]).reset_index(drop=True)
df.to_csv("/content/drive/MyDrive/dataset/labeled_tweets.csv", index=False)
```

So now we have collected our training data through distant supervision. In the next post, we will walk through the steps of text preprocessing and word embedding, then use it to build a Long Short-term Memory (LSTM) model. Stay tuned!

### Relevant Links

Project Github: [github.com/tmtsmrsl/TwitterSentimentAnalyzer](https://github.com/tmtsmrsl/TwitterSentimentAnalyzer)  
Streamlit App: [twitter-sentiment.streamlit.app/](https://twitter-sentiment.streamlit.app/)
