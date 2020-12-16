---
date: 2020-12-02T11:15:58-04:00
description: "Natural Language Processing"
featured_image: ""
tags: []
title: "I Tried Analysing Trump's Speeches"
---

### And this is how it turned out

It all started when I saw a corpus of transcribed pre-election rally speeches by Donald Trump. It sparked an interest in me to run a short overall analysis of it not going into many details or listening to each one of them (that would be *tough*, to say the least). For anyone interested, the corpus can be found on [Kaggle](https://www.kaggle.com/christianlillelund/donald-trumps-rallies)






For the analysis, I mainly used Natural Language Processing Toolkit libraries.

```python
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.collocations import *
import matplotlib.pyplot as plt
import gensim
from wordcloud import WordCloud

rallies = PlaintextCorpusReader('data/rallies', '.*')
stopwords = nltk.corpus.stopwords.words('english')
```
First I decided what I was going to look into and, accordingly, I made some functions that would help me with that. 
The first is a simple collocation finder, then - a word normalizing function (you know, the usual, excluding stopwards, numericals.. And next, I made a function for calculating the **text comprehension coefficient**.


```python
def colloc(words, n, num):
    if n == 2:
        collocation_finder = BigramCollocationFinder.from_words(words)
        ngram_measures = nltk.collocations.BigramAssocMeasures()
    elif n == 3:
        collocation_finder = TrigramCollocationFinder.from_words(words)
        ngram_measures = nltk.collocations.TrigramAssocMeasures()
    elif n == 4:
        collocation_finder = QuadgramCollocationFinder.from_words(words)
        ngram_measures = nltk.collocations.QuadgramAssocMeasures()
    else:
        return []
    
    collocation_finder.apply_freq_filter(3)
    
    def filter(word):
        return not word.isalpha()
    
    collocation_finder.apply_word_filter(filter)
    
    return collocation_finder.nbest(ngram_measures.pmi, num)
```


```python
def normalize_words(corpus, stop):
    lists_of_words = [] 
    for fileid in rallies.fileids(): 
        words = rallies.words(fileid)
        words = [word.lower() for word in words if word[0].isalpha() and word.lower() not in stop]
        lists_of_words.append(words)
    return lists_of_words
```


```python
def TCC(words, sentences): #text comprehension coefficient
    A = len(words)
    B = len(sentences)
    C = 0
    for word in words:
        if len(word) > 6:
            C += 1
    readability = round(A / B + C / A * 100)
    return readability

def CL(readability): #comprehension level   
    if readability <= 24:
        return "very easy"
    elif readability >= 25 and readability <= 34:
        return "easy"
    elif readability >= 35 and readability <= 44:
        return "moderately difficult"
    elif readability >= 45 and readability <= 54:
        return "difficult"
    else:
        return "very difficult"
```

First, let's look at some of his most commonly used n-grams.


```python
words = list(rallies.words())

print('Most common bi-grams:', colloc(words, 2, 10))
print('Most common tri-grams:', colloc(words, 3, 10))
print('Most common quad-grams:', colloc(words, 4, 10))
```

    Most common bi-grams: [('Gavin', 'Newsom'), ('OPEC', 'Plus'), ('Soo', 'Locks'), ('concentration', 'camps'), ('medieval', 'style'), ('shining', 'oasis'), ('whistle', 'blower'), ('yellow', 'vests'), ('Carolyn', 'Maloney'), ('Des', 'Moines')]
    Most common tri-grams: [('Operation', 'Warp', 'Speed'), ('ice', 'skating', 'rink'), ('Historically', 'Black', 'Colleges'), ('flawless', 'precision', 'strike'), ('initiatives', 'combating', 'kidney'), ('Osama', 'bin', 'Laden'), ('Wall', 'Street', 'Journal'), ('historically', 'black', 'colleges'), ('Paris', 'Climate', 'Accord'), ('Coast', 'Guard', 'cutters')]
    Most common quad-grams: [('Under', 'Operation', 'Warp', 'Speed'), ('initiatives', 'combating', 'kidney', 'disease'), ('Marine', 'Corps', 'Air', 'Station'), ('an', 'ice', 'skating', 'rink'), ('National', 'Border', 'Patrol', 'Council'), ('winter', 'at', 'Valley', 'Forge'), ('new', 'initiatives', 'combating', 'kidney'), ('actively', 'planning', 'new', 'attacks'), ('disastrous', 'Iran', 'Nuclear', 'Deal'), ('mobilization', 'since', 'World', 'War')]


Some of which are very topical, while some... hmm. OK, let's check the overall comprehension coefficient.
> Mind you, \
> 54 means very difficult; \
> 45 - 54 is difficult \
> 35 - 44 moderately difficult \
> 25 - 34 easy \
> < 24 - very, very easy. Like kindergarten-easy.


```python
print('Text Comprehension Coefficient')

sentences = rallies.sents()
coefficient = TCC(words, sentences)
comprehension = CL(coefficient)
print(f'Overall average score: {coefficient}, {comprehension}')
```

    Text Comprehension Coefficient
    Overall average score: 24, very easy


Just for a slight comparison, usually presidential speeches range anywhere between 34-60. But we all know there is nothing usual about Trump. \
\
Finally, I wanted to try out topic modelling solely to see what are Trump's go-to points in these pre-election rallies. This one was very interesting. I used Latent Dirichlet allocation for the modelling, and the rationale for such no_below and no_above parameters was to exclude all the State names that were mentioned (as they were mentioned *a lot*).


```python
lists_of_words = normalize_words(rallies, stopwords)
dictionary = gensim.corpora.Dictionary(lists_of_words)

dictionary.filter_extremes(no_below=14, no_above=0.5)
print('Number of unique words: %d' % len(dictionary))

corpus_bow = [dictionary.doc2bow(words) for words in lists_of_words]

num_topics = 10

# topic modeling using Latent Dirichlet allocation
lda = gensim.models.LdaMulticore(
    corpus_bow,            
    id2word=dictionary,    
    num_topics=num_topics, 
    iterations=200,        
    passes=20,             
    workers=3,             
    chunksize=1000,        
    eval_every=None        
    )

print(lda)
```

    Number of unique words: 267
    LdaModel(num_terms=267, num_topics=10, decay=0.5, chunksize=1000)


Here you can check it out for yourself. Do these topics make sense solely judging by the most common keywords? Not really. But can we observe some patterns? Definitely. A lot of words with negative connotation were used, as they usually tend to attract more attention.


```python
for i in range(0, lda.num_topics):
    terms = [term for term, val in lda.show_topic(i, 10)]
    print('Topic #' + str(i) + ':', ', '.join(terms))
```

    Topic #0: john, dan, presidential, water, failed, easier, impossible, kavanaugh, corrupt, car
    Topic #1: virus, sent, stopping, weak, iraq, unless, despite, playing, ill, statement
    Topic #2: difference, lucky, someday, rather, ratings, crowds, fun, strike, picked, doctors
    Topic #3: anti, statement, dan, convention, achieved, peace, setting, twice, virus, interesting
    Topic #4: corrupt, ukraine, scott, father, statement, past, defeated, swamp, born, failed
    Topic #5: points, nine, peace, ratings, presidents, mistake, stories, black, presidential, exciting
    Topic #6: dan, virus, peace, car, john, troops, enthusiasm, list, minister, mistake
    Topic #7: water, car, statement, crowds, swamp, steel, previous, pressure, attorney, write
    Topic #8: steel, iowa, released, crimes, alien, assault, ill, kidding, voting, virginia
    Topic #9: iowa, swamp, arena, pouring, average, immigrants, alien, zones, die, sold


And here are some pretty wordclouds for those that are already tired of reading this.


```python
for topic in range(lda.num_topics):
    plt.figure()
    
    wordcloud = WordCloud(background_color='white')      # creating wordcloud object
    words_with_weights = dict(lda.show_topic(topic, 50)) # getting top 50 words for the topic
    wordcloud.fit_words(words_with_weights)              # distributing the words in a cloud
    plt.imshow(wordcloud)                                # showing the cloud
    
    plt.axis('off')
    plt.title('Topic #' + str(topic))
    plt.show()
```


    
![jpg](/output_15_0.jpg)




    
![jpg](/output_15_1.jpg)
   



    
![jpg](/output_15_2.jpg)
     



    
![jpg](/output_15_3.jpg)
      



    
![jpg](/output_15_4.jpg)
     



    
![jpg](/output_15_5.jpg)
      



    
![jpg](/output_15_6.jpg)
    



    
![jpg](/output_15_7.jpg)
    



    
![jpg](/output_15_8.jpg)
    



    
![jpg](/output_15_9.jpg)
    



```python
%matplotlib inline

n = 10 # how many top words we want to see

for i in range(0, lda.num_topics): # a loop through all topics
    plt.figure(figsize=(5, 3))
    
    words = [word for word, weight in lda.show_topic(i, n)] # taking the list of top n words
    weights = [weight for word, weight in lda.show_topic(i, n)] # taking the list of weights for the top n words
    
    plt.barh(range(0, n), weights) # plotting a horizontal bar chart
    
    # Adding title for the chart, text for x-axis label, and customizing y-axis ticks
    plt.title('Word weights for topic #' + str(i))
    plt.xlabel('Coefficient')
    plt.yticks(range(0, n), words)
    plt.ylim(n - 0.5, -0.5) # reversing y-axis and subtracting offset to make the chart more clear
    
    plt.show()
```


    
![jpg](/output_16_0.jpg)
    



    
![jpg](/output_16_1.jpg)
    



    
![jpg](/output_16_2.jpg)
    



    
![jpg](/output_16_3.jpg)
    



    
![jpg](/output_16_4.jpg)
    



    
![jpg](/output_16_5.jpg)
    



    
![jpg](/output_16_6.jpg)
    



    
![jpg](/output_16_7.jpg)
    



    
![jpg](/output_16_8.jpg)
    



    
![jpg](/output_16_9.jpg)
