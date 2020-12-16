---
title: "Sentiment Analysis"
date: 2020-12-16T12:46:48+02:00
draft: false
description: "Sentiment Analysis"
title: "I Made A Machine Learning Model For Analysing Movie Reviews So You Don't Have To"
---

### The Goal
Have you ever based your decision on whether to watch a movie or not on other people's reviews and feedback? I have. So naturally, I wanted to create a program that would help me decide. Thus, I created a machine learning model in Python for scalar/degree sentiment analysis to train labeling positive versus negative film reviews, by analysing their features and adjectives, in case anyone wants to use it on a corpus of any kind of film reviews (it might not be so accurate on other types of corpora, as I've only trained it on films, but you might try!). For this, I used a set of corpus of 10000 IMDB film reviews. The full corpus is available [here]( http://ai.stanford.edu/~amaas/data/sentiment/)
I followed a tutorial from TowardsDataScience with several tweaks and adaptations, the original article is available [here](https://towardsdatascience.com/imdb-reviews-or-8143fe57c825); however, as opposed to this example, I only used 3 classifiers and tweaked them in such a way, that surprisingly yielded very optimal results. As my IDE, I used PyCharm.
### Chosen Approach
I have chosen to try three classifying algorithms - original Naive Bayes, Logistic Regression and Support Vector classifiers. Then I provided their f1 accuracy scores.
### Needed Results
1. Normalized bag of words (BOW) of the corpus;
2. WordCloud for most positive adjectives found in the corpus;
3. Data split into training set and testing set;
4. Metrics for each chosen classifier;
5. Most informative features with trained over the positive and negative words in the reviews.
### Result Evaluation
The training was really successful and all three of the chosen classifiers have given rather high accuracy score, taking into account that Sentiment Analyses usually don't yield as high accuracy as statistical analyses. For this reason, anything around 70% is a very good score, while in my case, the training models all achieved precision ranging from 77-80%. Yay! Now it can be used to classify any other texts!
### Visualizations of My Results
1. Figure 1: Wordcloud of positive words found in the corpus;
![wordcloud](/Figure_1_wordcloud.jpg)
2. Figure 2: Confusion matrix non-normalized;
![confusion_matrix](/Figure2_Confusion_matrix_not_normalized.jpg)
3. Figure 3: Confusion matrix normalized;
![confusion_matrix](/Figure3_Confusion_matrix_normalized.jpg)



