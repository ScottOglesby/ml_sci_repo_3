from __future__ import print_function

import argparse
import operator
import pyspark

from parallelm.mlops import mlops as mlops


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--words-file", help="Path of the file whose words need be counted")
    options = parser.parse_args()
    return options


def count_words(sc, words_file):
    lines = sc.textFile(words_file)
    words = lines.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1))
    counts = words.reduceByKey(operator.add)
    sorted_counts =  counts.sortBy(lambda x: x[1], False)
    total_words = sorted_counts.count()
    mlops.set_stat("total_words_1_push_2", total_words)

    total_words = 0
    for word,count in sorted_counts.toLocalIterator():
        print(u"{} --> {}".format(word, count))
        total_words += 1
    mlops.set_stat("total_words_2_push_2", total_words)


def main():

    sc = pyspark.SparkContext(appName="PySparkWordCount")
    mlops.init(sc)

    options = parse_args()
    if options.words_file is None:
        print("Bad option - no file was provided")
        mlops.set_stat("got_file", 0)
    else:
        mlops.set_stat("got_file", 1)
        count_words(sc, options.words_file)

    sc.stop()
    mlops.done()

if __name__ == "__main__":
    main()

