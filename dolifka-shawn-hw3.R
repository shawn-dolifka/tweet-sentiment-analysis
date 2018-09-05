#Shawn Dolifka
#CSCE 489-700

#Code for "no tweet_id" is pointless. Ran out of time trying to debug
#and didn't delete

library(tidyverse)
library(tidytext)
library(caret)
library(e1071)

#Import tweets into tibble
tweets = read_csv("sentiment.self.drive.csv")

#Second set of Tweets without tweet_id
tweets2 = tweets %>%
  select(-one_of("tweet_id"))

#==========================================================#
#Naive code
#==========================================================#

#Turn tibble data into tokens
tokenTweets = tweets %>% unnest_tokens(word,text)

#Remove stop words from data
stopTweets = tokenTweets %>% anti_join(stop_words)

#Track total number of Tweets to find missing Tweets after inner_join
index = tibble(
  tweet_id = 1:2385
)

#Calculate actual value totals
actualPosTotal = sum(tweets$sentiment == "positive")
actualNeutTotal = sum(tweets$sentiment == "neutral")
actualNegTotal = sum(tweets$sentiment == "negative")

actualPos = rep("positive", actualPosTotal)
actualNeut = rep("neutral", actualNeutTotal)
actualNeg = rep("negative", actualNegTotal)

reference = c(actualPos, actualNeut, actualNeg)

#-------------------------------------------------#
#Helper Functions
#-------------------------------------------------#

#Function to calculate accuracy
accuracy.calc <- function(x)
{
  acc = sum(diag(x)) / sum(x)
  return(acc)
}

#Function to calculate precision
precision.calc <- function(x)
{
  prec = diag(x) / rowSums(x)
  return(prec)
}

#Function to calculate recall
recall.calc <- function(x)
{
  rec = diag(x) / colSums(x)
  return(rec)
}

#Function to calculate F1
f1.calc <- function(x)
{
  f1 = ifelse(precision.calc(x) + recall.calc(x) == 0, 0,
         (2 * precision.calc(x) * recall.calc(x)) / 
           (precision.calc(x) + recall.calc(x)))
  return(f1)
}

#-------------------------------------------------#
#Afinn Method
#-------------------------------------------------#

#Perform an afinn sentiment analysis on Tweets
afinn <- stopTweets %>% inner_join(get_sentiments("afinn")) %>%
  group_by(index = tweet_id) %>%
  summarise(sentiment = sum(score)) %>%
  mutate(method = "AFINN") %>%
  rename(tweet_id = index)

#Fill in all Tweets filtered out by inner_join as neutral
antiTweet = index %>% anti_join(afinn, by = "tweet_id") %>%
  mutate(method = "AFINN", sentiment =0)
afinn = afinn %>% full_join(antiTweet) %>% arrange(tweet_id)

#Classify Tweets as positive, negative, neutral
afinn = afinn %>% mutate(
  sent = ifelse(sentiment > 0, "positive",
                ifelse(sentiment == 0, "neutral","negative"))
)

#Build the confusion matrix
predictPosTotal = sum(afinn$sent == "positive")
predictNeutTotal = sum(afinn$sent == "neutral")
predictNegTotal = sum(afinn$sent == "negative")

predictPos = rep("positive", predictPosTotal)
predictNeut = rep("neutral", predictNeutTotal)
predictNeg = rep("negative", predictNegTotal)

predicted = c(predictPos, predictNeut, predictNeg)

confuMatrix = table(predicted,reference)

#Calculate the Precision, Recall and F1
afinnPrecision = precision.calc(confuMatrix)
afinnRecall = recall.calc(confuMatrix)
afinnF1 = f1.calc(confuMatrix)

#-------------------------------------------------#
#Bing Method
#-------------------------------------------------#

#Bing sentiment analysis
bing = stopTweets %>% inner_join(get_sentiments("bing")) %>% 
  mutate(method = "Bing et al.")%>%
  count(method, index = tweet_id, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  rename(tweet_id = index)

#Fill in all Tweets filtered out by inner_join as neutral
antiTweet = index %>% anti_join(bing, by = "tweet_id") %>%
  mutate(method = "Bing et al.", negative = 0, positive = 0, sentiment =0)
bing = bing %>% full_join(antiTweet) %>% arrange(tweet_id)

#Classify Tweets as positive, negative, neutral
bing = bing %>% mutate(
  sent = ifelse(sentiment > 0, "positive",
                ifelse(sentiment == 0, "neutral","negative"))
)

#Build the confusion matrix
predictPosTotal = sum(bing$sent == "positive")
predictNeutTotal = sum(bing$sent == "neutral")
predictNegTotal = sum(bing$sent == "negative")

predictPos = rep("positive", predictPosTotal)
predictNeut = rep("neutral", predictNeutTotal)
predictNeg = rep("negative", predictNegTotal)

predicted = c(predictPos, predictNeut, predictNeg)

confuMatrix = table(predicted,reference)

#Calculate the Precision, Recall and F1
bingPrecision = precision.calc(confuMatrix)
bingRecall = recall.calc(confuMatrix)
bingF1 = f1.calc(confuMatrix)

#-------------------------------------------------#
#NRC Method
#-------------------------------------------------#

#NRC sentiment analysis
nrc = stopTweets %>% 
  inner_join(get_sentiments("nrc") %>%
  filter(sentiment %in% c("positive","negative"))) %>%
  mutate(method = "NRC") %>% count(method, index = tweet_id, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  rename(tweet_id = index)

#Fill in all Tweets filtered out by inner_join as neutral
antiTweet = index %>% anti_join(nrc, by = "tweet_id") %>%
  mutate(method = "NRC", negative = 0, positive = 0, sentiment =0)
nrc = nrc %>% full_join(antiTweet) %>% arrange(tweet_id)

#Classify Tweets as positive, negative, neutral
nrc = nrc %>% mutate(
  sent = ifelse(sentiment > 0, "positive",
                ifelse(sentiment == 0, "neutral","negative"))
)

#Build the confusion matrix
predictPosTotal = sum(nrc$sent == "positive")
predictNeutTotal = sum(nrc$sent == "neutral")
predictNegTotal = sum(nrc$sent == "negative")

predictPos = rep("positive", predictPosTotal)
predictNeut = rep("neutral", predictNeutTotal)
predictNeg = rep("negative", predictNegTotal)

predicted = c(predictPos, predictNeut, predictNeg)

confuMatrix = table(predicted,reference)

#Calculate the Precision, Recall and F1
nrcPrecision = precision.calc(confuMatrix)
nrcRecall = recall.calc(confuMatrix)
nrcF1 = f1.calc(confuMatrix)


#==========================================================#
##TF-IDF code
#==========================================================#

#Separate Tweets out by sentiment
positiveTweets = subset(tweets,sentiment == "positive")
neutralTweets = subset(tweets,sentiment == "neutral")
negativeTweets = subset(tweets,sentiment == "negative")

#Separate Tweets out by sentiment, no tweet_id
positiveTweets = subset(tweets2,sentiment == "positive")
neutralTweets = subset(tweets2,sentiment == "neutral")
negativeTweets = subset(tweets2,sentiment == "negative")

#-------------------------------------------------#
#-------------------------------------------------#

#Get Tweet words, no tweet_id version
tweetWords2 = tweets2 %>%
  unnest_tokens(word,text) %>% 
  anti_join(stop_words) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()

#Get Positive Tweet words, no tweet_id version
pos_tweetWords = positiveTweets %>%
  unnest_tokens(word,text) %>% 
  anti_join(stop_words) %>%
  count(word, sort = TRUE) %>%
  ungroup()

#Get Neutral Tweet words, no tweet_id version
neut_tweetWords = neutralTweets %>%
  unnest_tokens(word,text) %>% 
  anti_join(stop_words) %>%
  count(word, sort = TRUE) %>%
  ungroup()

#Get Negative Tweet words, no tweet_id version
neg_tweetWords = negativeTweets %>%
  unnest_tokens(word,text) %>% 
  anti_join(stop_words) %>%
  count(word, sort = TRUE) %>%
  ungroup()

#-------------------------------------------------#
#-------------------------------------------------#

#Get Tweet words
tweetWords = tweets %>%
  unnest_tokens(word,text) %>% 
  anti_join(stop_words) %>%
  count(word, tweet_id, sentiment, sort = TRUE) %>%
  ungroup()

#Get Positive Tweet words
pos_tweetWords = positiveTweets %>%
  unnest_tokens(word,text) %>% 
  anti_join(stop_words) %>%
  count(word, tweet_id, sort = TRUE) %>%
  ungroup()

#Get Neutral Tweet words
neut_tweetWords = neutralTweets %>%
  unnest_tokens(word,text) %>% 
  anti_join(stop_words) %>%
  count(word, tweet_id, sort = TRUE) %>%
  ungroup()

#Get Negative Tweet words
neg_tweetWords = negativeTweets %>%
  unnest_tokens(word,text) %>% 
  anti_join(stop_words) %>%
  count(word, tweet_id, sort = TRUE) %>%
  ungroup()

#-------------------------------------------------#
#-------------------------------------------------#

#Get total num Tweet words
totalWords = tweetWords %>%
  group_by(tweet_id)%>%
  summarize(total = sum(n))

#Get total num Positive Tweet words
pos_totalWords = pos_tweetWords %>%
  group_by(tweet_id)%>%
  summarize(total = sum(n))

#Get total num Neutral Tweet words
neut_totalWords = neut_tweetWords %>%
  group_by(tweet_id)%>%
  summarize(total = sum(n))

#Get total num Negative Tweet words
neg_totalWords = neg_tweetWords %>%
  group_by(tweet_id)%>%
  summarize(total = sum(n))

#-------------------------------------------------#
#-------------------------------------------------#

#Get total num Tweet words, no tweet_id version
totalWords = tweetWords %>%
  group_by(word)%>%
  summarize(total = sum(n))

#Get total num Positive Tweet words, no tweet_id version
pos_totalWords = pos_tweetWords %>%
  group_by(word)%>%
  summarize(total = sum(n))

#Get total num Neutral Tweet words, no tweet_id version
neut_totalWords = neut_tweetWords %>%
  group_by(word)%>%
  summarize(total = sum(n))

#Get total num Negative Tweet words, no tweet_id version
neg_totalWords = neg_tweetWords %>%
  group_by(word)%>%
  summarize(total = sum(n))

#-------------------------------------------------#
#-------------------------------------------------#

#Combine all words
tweetWords = left_join(tweetWords, totalWords)

#Combine Positive words
pos_tweetWords = left_join(pos_tweetWords, pos_totalWords)

#Combine Neutral words
neut_tweetWords = left_join(neut_tweetWords, neut_totalWords)

#Combine Negative words
neg_tweetWords = left_join(neg_tweetWords, neg_totalWords)

#-------------------------------------------------#
#-------------------------------------------------#

#Perform TF-IDF all words
tfidfTweets = tweetWords %>% bind_tf_idf(word, tweet_id, n) %>%
  arrange(desc(tf_idf))

#Perform TF-IDF all words, no tweet_id
tfidfTweets = tweetWords2 %>% bind_tf_idf(word, sentiment, n) %>%
  arrange(desc(tf_idf))

#Clean up TF-IDF All Words results
tfidfTweets = tfidfTweets %>%
  #filter(tf_idf>=0.0001478617)
  filter(tf_idf>=0.7)

#Perform TF-IDF Positive words
pos_tfidfTweets = pos_tweetWords %>% bind_tf_idf(word, tweet_id, n) %>%
  arrange(desc(tf_idf))

#Clean up TF-IDF Positive results
pos_tfidfTweets = pos_tfidfTweets %>% select(-one_of("tweet_id")) %>%
  select(-one_of("n")) %>%
  select(-one_of("total")) %>%
  select(-one_of("tf")) %>%
  select(-one_of("idf")) %>%
  mutate(sentiment = "positive") %>%
  filter(tf_idf>=0.7)

#Perform TF-IDF Neutral words
neut_tfidfTweets = neut_tweetWords %>% bind_tf_idf(word, tweet_id, n) %>%
  arrange(desc(tf_idf))

#Clean up TF-IDF Neutral results
neut_tfidfTweets = neut_tfidfTweets %>% select(-one_of("tweet_id")) %>%
  select(-one_of("n")) %>%
  select(-one_of("total")) %>%
  select(-one_of("tf")) %>%
  select(-one_of("idf")) %>%
  mutate(sentiment = "neutral") %>%
  filter(tf_idf>=0.7)

#Perform TF-IDF Negative words
neg_tfidfTweets = neg_tweetWords %>% bind_tf_idf(word, tweet_id, n) %>%
  arrange(desc(tf_idf))

#Clean up TF-IDF Negative results
neg_tfidfTweets = neg_tfidfTweets %>% select(-one_of("tweet_id")) %>%
  select(-one_of("n")) %>%
  select(-one_of("total")) %>%
  select(-one_of("tf")) %>%
  select(-one_of("idf")) %>%
  mutate(sentiment = "negative") %>%
  filter(tf_idf>=0.7)

#Combine all terms to one tibble
combined_tfidf = pos_tfidfTweets %>%
  full_join(neut_tfidfTweets) %>% 
  full_join(neg_tfidfTweets)

#-------------------------------------------------#
#-------------------------------------------------#

train = sample(200,100)

#Try single TF-IDF model
mymodel = svm(sentiment~.,data=tfidfTweets, type = "C-classification")
prediction = predict(mymodel, tfidfTweets)
single_table = table(Predicted = prediction, Actual = tfidfTweets$sentiment)
f1.calc(single_table)
single_table

#Try combined TF-IDF model
mymodel = svm(sentiment~.,data=combined_tfidf, type = "C-classification")
prediction = predict(mymodel, combined_tfidf)
combine_table = table(Predicted = prediction, Actual = combined_tfidf$sentiment)
f1.calc(combine_table)

precision.calc(single_table)
recall.calc(single_table)
f1.calc(single_table)

set.seed(123)
tune.out = tune(svm, type = "C-classification", sentiment~., data = tfidfTweets,
                kernel="radial",
                ranges=list(cost=1^(-1:2),
                gamma=c(1.0,1:3))
                )

