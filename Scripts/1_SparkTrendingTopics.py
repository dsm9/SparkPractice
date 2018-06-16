import os, shutil
import sys
import json
import unicodedata
import pyspark 
from operator import add
from time import time

def execute(inputFile, outputTrending, outputTopN, numPartitions):
    print "1_SparkTrendingTopics.py"
    print "    inputFile: ", inputFile
    print "    outputTrending: ", outputTrending
    print "    outputTopN: ", outputTopN
    print "    numPartitions: ", numPartitions
    
    t0 = time()
    sc = pyspark.SparkContext('local[*]')

#    inputFile = "../Datasets/Tweets/tweets_es.json"
#    outputTrending = "../Results/Trending"
#    outputTopN = "../Results/TopN"

    # Trending topics

    input = sc.textFile(inputFile, int(numPartitions))
    tweets = input.map(lambda x: json.loads(x))
    print "* Total tweets: ", tweets.count()

    tweets_es = tweets.filter(lambda t: "es" in t["lang"])
    print "* Spanish tweets: ", tweets_es.count()

    tweets_es_hashtags = tweets_es.filter(lambda t: t["entities"]["hashtags"] != [] )
    print  "* Tweets with hashtags: ", tweets_es_hashtags.count()

    hashtags = tweets_es_hashtags.flatMap(lambda t: map(lambda h: (unicodedata.normalize('NFKD', h["text"]).encode('ascii','ignore'),1), t["entities"]["hashtags"]))
    print "* Hashtags: ", hashtags.count()

    trending_hashtags = hashtags.reduceByKey(lambda a, b: a + b)
    print "* Hashtags reduced: ", trending_hashtags.count()

    if os.path.exists(outputTrending): 
        shutil.rmtree(outputTrending)
    trending_hashtags.saveAsTextFile(outputTrending)
    print "* Files saved: '" + outputTrending + "'"
    print "* Some examples: ", trending_hashtags.take(10)

    # TopN
    trending_sorted = trending_hashtags.takeOrdered(10, key=lambda t: -t[1])
    print "* Top N: ", trending_sorted

    if os.path.exists(outputTopN): 
        shutil.rmtree(outputTopN)
    sc.parallelize(trending_sorted).saveAsTextFile(outputTopN)
    print "* Files saved: '" + outputTopN + "'"

    # SENTIMENT
    print ""
    print "* Hashtag sentiment"
    file_positive = sc.textFile("../Dictionary/positive_words_es.txt")
    file_negative = sc.textFile("../Dictionary/negative_words_es.txt")
    positive_words = file_positive.map(lambda w: w.encode('ascii', 'ignore'))
    negative_words = file_negative.map(lambda w: w.encode('ascii', 'ignore'))
    positive_words_list = positive_words.collect()
    negative_words_list = negative_words.collect()

    print "  Positive words: ", positive_words.take(10), "..."
    print "  Negative words: ", negative_words.take(10), "..."
    
    tweets_hashtags = tweets_es_hashtags.map(lambda t: (unicodedata.normalize('NFKD',t["text"]).encode('ascii','ignore').lower(),\
                                        (map(lambda h: unicodedata.normalize('NFKD',h["text"]).encode('ascii','ignore').lower(), \
                                        t["entities"]["hashtags"]))))\
                                        .flatMapValues(lambda x: x)
    print "  Tweets and hashtags: ", tweets_hashtags.take(2)
    
    def HashtagSentiment(tweet):
        positive_count = 0
        negative_count = 0
        hashtag = tweet[1]
        words = tweet[0].split(" ")
        length = len(words)
        for word in words:
            if word in positive_words_list:
                positive_count += 1
            elif word in negative_words_list:
                negative_count += 1

        return (hashtag, (length, positive_count, negative_count))

    hashtags_base_sentim = tweets_hashtags.map(lambda t: HashtagSentiment(t))\
                    .reduceByKey(lambda a, b:(a[0]+b[0], a[1]+b[1], a[2]+b[2]))
    print "  Hasghtag base sentiment info: "
    print hashtags_base_sentim.take(10)

    hashtags_sentim = hashtags_base_sentim.map(lambda h: (h[0],float(h[1][1]-h[1][2])/h[1][0]))
    print ""
    print "* Hashtag sentiments: "
    print hashtags_sentim.take(10)
    
    tt = time() - t0
    print ""
    print("* Completed in {} seconds".format(round(tt,3)))

if __name__ == '__main__':
    if len(sys.argv) == 1:
        #    print ("Usage -> python 1_SparkTrendingTopics.py inputFile outputTrending outputTopN numPartitions")
        inputFile = "../Datasets/Tweets/tweets_es.json"
        outputTrending = "../Results/Trending"
        outputTopN = "../Results/TopN"    
        numPartitions = 3
    else:
        inputFile = sys.argv[1]
        outputTrending = sys.argv[2]
        outputTopN = sys.argv[3]
        numPartitions = sys.argv[4]
    execute(inputFile, outputTrending, outputTopN, numPartitions)
        