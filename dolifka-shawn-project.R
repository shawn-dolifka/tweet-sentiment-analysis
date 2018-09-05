#Shawn Dolifka
#CSCE 489-700

library(tidyverse)
library(tidytext)
library(e1071)
library(caret)
library(reshape2)

#-------------------------------------------------#
#Helper Functions
#-------------------------------------------------#

#Shannon entropy function
shannon.entropy = function(p, base = 2)
{
  # remove zeros and normalize, just in case
  p = p[p > 0] / sum(p)
  
  H = sum(-p*log(p,base))
  
  # return the value of H
  return(H)
}

#Normalize the probabilities
normalize = function(x)
{
  x = x/sum(x)
  return(x)
}

#-------------------------------------------------#
#Import the data and clean it
#-------------------------------------------------#

#Import tweets into tibble 
#Rename input data so I can copy and paste my code
tweets = read_csv("hate.speech.csv") %>% 
  rename(text = tweet_text) %>%
  rename(sentiment = speech)

#Mutate the tweet_id for easier use
tweets = tweets %>% mutate(tweet_id = 1:nrow(tweets))

#Remove Twitter handles
tweets$text = gsub("(^|[^@\\w])@(\\w{1,15})\\b","",tweets$text)

#Remove URLs
tweets$text = gsub("\\s?(f|ht)(tp)(s?)(://)([^\\.]*)[\\.|/](\\S*)","",tweets$text)

#Remove numbers
tweets$text = gsub('[[:digit:]]+', '', tweets$text)

#Remove random punctuation
tweets$text = gsub("(?!')[[:punct:]]", "", tweets$text, perl=TRUE)

#Track total number of Tweets to find missing Tweets after inner_join
#The word will be filled in with "the" as a placeholder
index = tibble(
  tweet_id = 1:nrow(tweets),
  sentiment = "regular",
  word = "the"
)

#-------------------------------------------------#
#Tokenize Words
#-------------------------------------------------#

#Get Tweet words
tweetWords = tweets %>%
  unnest_tokens(word,text) %>% 
  anti_join(stop_words)

#Find the tweet_ids that contained only stop words 
anti_tweetWords = anti_join(index,tweetWords, by = "tweet_id")

#Join both tweets so that none are missing
tweetWords = full_join(anti_tweetWords,tweetWords)

#==========================================================#
#Code to create data for First Set of Features
#==========================================================#

#-------------------------------------------------#
#Perform sentiment analysis, Afinn Method
#-------------------------------------------------#

#Perform an afinn sentiment analysis on Tweets
afinn <- tweetWords %>% inner_join(get_sentiments("afinn")) %>% 
  mutate(sentiment = ifelse(score >= 0, "positive","negative"))

anti_afinn = tweetWords %>% 
  anti_join(get_sentiments("afinn")) %>% 
  mutate(score = 0, sentiment = "positive")

afinn = full_join(afinn, anti_afinn)

#-------------------------------------------------#
#Calculate word frequencies
#-------------------------------------------------#

#Word frequency for all Tweets combined
tweet_freq = afinn %>% 
  group_by(tweet_id, sentiment, score) %>%
  count(word, sort = T) %>%
  ungroup() %>%
  mutate(probability = n / sum(n)) %>%
  select(-n)

#Split up Tweets by tweet_id
new_freq = split(tweet_freq,tweet_freq$tweet_id)

#Normalize the probability of all individual Tweets
for (i in 1:length(new_freq))
{
  new_freq[[i]]$probability = normalize(new_freq[[i]]$probability)
  
  #Append the entropy to the Tweets
  new_freq[[i]] = new_freq[[i]] %>%
    mutate(entropy = shannon.entropy(new_freq[[i]]$probability))
  
  new_freq[[i]] = new_freq[[i]] %>%
    mutate(positive = sum(new_freq[[i]]$score[which(new_freq[[1]]$score > 0)]))
  
  new_freq[[i]] = new_freq[[i]] %>% 
    mutate(negative = sum(new_freq[[i]]$score[which(new_freq[[1]]$score < 0)]))
  
  new_freq[[i]]$negative = new_freq[[i]]$negative %>% replace_na(0)
}

#Bind all Tweets back together into original Tibble
new_freq = do.call("rbind", new_freq)

#Combine all tweet_ids to a single value
new_freq = new_freq[!duplicated(new_freq$tweet_id), ] %>% select(-word, -score)

#Change the sentiments to a factor for the SVM
new_freq$sentiment = as.factor(new_freq$sentiment)

#Rename new_freq so I don't forget it
feature_one = new_freq %>% 
  select(-probability,-tweet_id)
rm(new_freq)

#==========================================================#
#Code to create data for Second Set of Features
#==========================================================#

#-------------------------------------------------#
#TF-IDF code
#-------------------------------------------------#

#Get Tweet words with a count of words
countWords = tweetWords %>%
  count(word, tweet_id, sentiment, sort = TRUE) %>%
  ungroup()

#Get total num Tweet words
totalWords = countWords %>%
  group_by(tweet_id)%>%
  summarize(total = sum(n))

#Combine all words
combineWords = left_join(countWords, totalWords)

#Perform TF-IDF all words
feature_two = combineWords %>% 
  bind_tf_idf(word, tweet_id, n) %>%
  arrange(desc(tf_idf)) %>%
  select(-n,-total,-tf,-idf,-tweet_id)

#==========================================================#
#Classification: First Feature Set
#==========================================================#

#Split the data into "training" (70%) and "validation" (30%)
#feature_one = matrix(as.numeric(unlist(feature_one)),nrow=nrow(feature_one))
set.seed(2)
independent = sample(2, nrow(feature_one), replace = TRUE, prob=c(0.7, 0.3))
trainset_1 = feature_one[independent == 1,]
testset_1 = feature_one[independent == 2,]

#-------------------------------------------------#
#Create the SVM
#-------------------------------------------------#

#Build a model for the SVM
model = svm(sentiment~., data = trainset_1, kernel="radial", cost=1, gamma
              = 1/ncol(trainset_1), scale = FALSE)

#Predict the labels of the testing data
svm.pred = predict(model, testset_1[, !names(testset_1) %in% c("sentiment")])

#Generate a classification table with the results
svm.table = table(svm.pred, testset_1$sentiment)

#Generate the confusion matrix
confusionMatrix(svm.table)

#-------------------------------------------------#
#Tune the SVM
#-------------------------------------------------#

#Tune the SVM
tuned = tune.svm(sentiment~., data = trainset_1, 
                 gamma = 10^(-6:-1), cost = 10^(1:2), scale = FALSE)

#Retrain the model with the best parameters
model.tuned = svm(sentiment~., data = trainset_1, gamma = 
                  tuned$best.parameters$gamma, 
                  cost = tuned$best.parameters$cost, scale = FALSE)

#Make prediction again with the best fit model
svm.tuned.pred = predict(model.tuned, testset_1[, !names(testset_1) %in%
                                                c("sentiment")])

#Generate a classification table for the best fit model
svm.tuned.table=table(svm.tuned.pred, testset_1$sentiment)

#Generate the confusion matrix for tuned model
confusionMatrix(svm.tuned.table)