from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import json

from gensim import corpora, models
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim

import re
import pandas as pd
import pathlib

# Your MongoDB Atlas connection string
WeiboDataset1_CONNECTION_STR = "mongodb+srv://:@weibodataset1.3fbrcoi.mongodb.net/?retryWrites=true&w=majority"
WeiboDataset2_CONNECTION_STR = "mongodb+srv://:@weibodataset2.bvjbm0q.mongodb.net/?retryWrites=true&w=majority"

CONNECTION_STR_LIST = {
    "weibodataset1": WeiboDataset1_CONNECTION_STR,
    "weibodataset2": WeiboDataset2_CONNECTION_STR,
}


def get_documents(db, collection_name, year, month):
    print("Get doc...")
    try:
        Database_client = MongoClient(
            CONNECTION_STR_LIST[db], server_api=ServerApi("1")
        )
        db = Database_client[db]
        collection = db[collection_name]

        if year is None and month is None:
            data_array = list(collection.find())
        else:
            if year is not None and month is None:
                start_date = datetime.strptime(
                    f"{year}-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"
                )
                end_date = datetime.strptime(
                    f"{year}-12-31T23:59:59Z", "%Y-%m-%dT%H:%M:%SZ"
                )
            elif year is not None and month is not None:
                formatted_start_month = str(month).zfill(2)
                end_month = int(month) + 1
                formatted_end_month = str(end_month).zfill(2)

                if end_month == 13:
                    start_date = datetime.strptime(
                        f"{year}-{formatted_start_month}-01T00:00:00Z",
                        "%Y-%m-%dT%H:%M:%SZ",
                    )
                    end_date = datetime.strptime(
                        f"{int(year) + 1}-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"
                    )
                else:
                    start_date = datetime.strptime(
                        f"{year}-{formatted_start_month}-01T00:00:00Z",
                        "%Y-%m-%dT%H:%M:%SZ",
                    )
                    end_date = datetime.strptime(
                        f"{year}-{formatted_end_month}-01T00:00:00Z",
                        "%Y-%m-%dT%H:%M:%SZ",
                    )

            query = {"created_at": {"$gte": start_date, "$lte": end_date}}
            data_array = list(collection.find(query))
        return [tweet["content_wordlist"] for tweet in data_array]

    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        Database_client.close()


def update_document(doc):
    try:
        Database_client = MongoClient(
            WeiboDataset1_CONNECTION_STR, server_api=ServerApi("1")
        )
        db = Database_client["weibodataset1"]
        collection = db["lda_json"]

        # Update the document with upsert option
        result = collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"lda_json": str(doc["lda_json"])}},
            upsert=True,  # Set upsert to true
        )

        print(f"Matched {result.matched_count} document(s).")
        if result.upserted_id:
            print(f"Inserted new document with _id: {result.upserted_id}")

    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        Database_client.close()


def gensim_lda_html(docs, filter=False, multicore=False, num_topics=5):
    def gensim_lda_auto_html(corpus, dictionary, multicore, num_topics):
        ldaModel = {}
        topic_list = {}
        Perplexity = {}
        for i in range(num_topics[0], num_topics[1] + 1):
            print("LDA/Training model...")
            if multicore:
                ldaModel[i] = models.LdaMulticore(
                    corpus=corpus,
                    id2word=dictionary,
                    chunksize=4000,
                    eta="auto",
                    iterations=400,
                    num_topics=i,
                    passes=20,
                    eval_every=None,
                    per_word_topics= True,
                )
            else:
                ldaModel[i] = models.LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    chunksize=4000,
                    alpha="auto",
                    eta="auto",
                    iterations=400,
                    num_topics=i,
                    passes=20,
                    eval_every=None,
                )
            print("LDA/print topic...")
            topic_list[i] = ldaModel[i].print_topics(i)
            print(i, ".")
            for j in range(0, len(topic_list[i])):
                print(topic_list[i][j])
            Perplexity[i] = ldaModel[i].log_perplexity(corpus)
            print("Perplexity: ", Perplexity[i])
            # coherence_model_lda = CoherenceModel(model=ldaModel[i], dictionary=dictionary, coherence='c_v')
            # coherence_lda = coherence_model_lda.get_coherence()
            # print('Coherence Score: ', coherence_lda)
            # print('')

        while True:
            i = int(input("Which one do you choose? [num] "))
            for j in range(0, len(topic_list[i])):
                print(topic_list[i][j])
            print("Perplexity: ", Perplexity[i])
            confirm = input("Confirm? (y/n) ")
            if confirm.lower() == "y":
                print("LDA/preparing show model...")
                prepare = pyLDAvis.gensim.prepare(ldaModel[i], corpus, dictionary)
                print("LDA/show prepared model to html...")
                htmlstring = pyLDAvis.prepared_data_to_html(prepare)
                return htmlstring  # Return the htmlstring

    def _gensim_lda_html(corpus, dictionary, multicore, num_topics):
        print("LDA/Training model...")
        if multicore:
            ldaModel = models.LdaMulticore(
                corpus=corpus,
                id2word=dictionary,
                chunksize=4000,
                eta="auto",
                iterations=400,
                num_topics=num_topics,
                passes=20,
                eval_every=None,
                per_word_topics= True,
            )
        else:
            ldaModel = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                chunksize=4000,
                alpha="auto",
                eta="auto",
                iterations=400,
                num_topics=num_topics,
                passes=20,
                eval_every=None,
            )
        print("LDA/preparing show model...")
        prepare = pyLDAvis.gensim.prepare(ldaModel, corpus, dictionary)
        print("LDA/show prepared model to html...")
        htmlstring = pyLDAvis.prepared_data_to_html(prepare)
        print("LDA/print topic...")
        topic_list = ldaModel.print_topics(num_topics)
        for i in range(0, len(topic_list)):
            print(topic_list[i])
        return htmlstring

    print("LDA...")

    dictionary = corpora.Dictionary(docs)

    dictionary.filter_n_most_frequent(200)
    if filter:
        dictionary.filter_extremes(no_below=20, no_above=0.5)

    dictionary.compactify()

    corpus = [dictionary.doc2bow(text) for text in docs]

    if type(num_topics) == list:
        return gensim_lda_auto_html(corpus, dictionary, multicore, num_topics)

    else:
        return _gensim_lda_html(corpus, dictionary, multicore, num_topics)


# Example usage def get_documents(db, collection_name, year, month):
if __name__ == "__main__":
    # weibodataset1:  metoo | girls help girls(hashtag)
    # weibodataset2: 米兔运动
    # get colleciton setting
    database = "weibodataset1"
    topic = r"girls help girls(hashtag)"
    year = 2022
    month = None
    num_topics = [3,6]
    # num_topics = 2

    # Gensim LDA setting
    filter = False
    multicore = True

    # run code

    try:
        if type(num_topics) == int:
            if num_topics <= 1:
                raise AssertionError
        if type(num_topics) == list:
            if num_topics[0] <= 1:
                raise AssertionError
        docs = get_documents(database, topic, year, month)
        lda_html = gensim_lda_html(
            docs, filter=filter, multicore=multicore, num_topics=num_topics
        )
        htmldoc = {
            "_id": topic + "_" + str(year) + "_" + str(month),
            "lda_json": lda_html,
        }
        upload = input("upload?(y/n)") == "y"
        print(upload)
        if upload:
            update_document(htmldoc)
    except ValueError:
        if filter:
            print("All topics are filtered, please try disable extreme filter")
        else:
            print("Cannot find Topic")
    except AssertionError:
        print("Minimum Topic num is 2")
