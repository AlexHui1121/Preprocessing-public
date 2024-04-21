import json
import os
import os.path
import glob
import re
import jieba
import jieba.posseg
import nltk
from snownlp import SnowNLP

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

yearMismatchCount = 0
JSONDecodeErrorCount = 0

# remove various punctuation marks, special characters, and emojis
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002500-\U00002BEF"
    "\U00002702-\U000027B0"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"  # dingbats
    "\u3030"
    "]+",
    flags=re.UNICODE,
)
pattern = re.compile(
    r'\t|\n|；|\.|。|：|：\.|-|:|\d|;|、|，|\)|\(|\?|"|“|”|/|（|）|\'|\"|！|？|,|<|>|@|#|\$|%|\^|&|\*|\||\\|_|`|~|\[|\]|{|}|\+|=|\u3000|，|。|、|；|‘|’|“|”|（|）|【|】|《|》|！|？|…|·|「|」|『|』|〈|〉|〔|〕|—|…|‧|﹏|︰|︱|︳|︴|︵|︶|︷|︸|︹|︺|︻|︼|︽|︾|︿|﹀|﹁|﹂|﹃|﹄|﹅|﹆|﹉|﹊|﹋|﹌|﹍|﹎|﹏'
)


def create_stopword():
    stopwords = []
    for file in os.listdir("stopword"):
        print(file)
        with open(rf"stopword\{file}", "r", encoding="utf-8") as fp:
            stopwords.extend(fp.read().split())
    return stopwords


def text_preprocessing(text):
    string_data = re.sub(pattern, "", text)
    string_data = re.sub(emoji_pattern, "", string_data)
    return string_data


def text_segmentation(string_data):
    # Text segmentation
    chinese_data = re.findall(r"[\u4e00-\u9fff]+", string_data)
    english_data = re.findall(r"[a-zA-Z\u00C0-\u00FF]+", string_data)
    chinese_object_list = []
    english_object_list = []

    # Chinese text segmentation using Jieba
    for sentence in chinese_data:
        seg_list_exact = jieba.cut(sentence, cut_all=False)
        # chinese_object_list.extend(seg_list_exact)
        chinese_object_list.extend(seg_list_exact)

    # English text segmentation using NLTK
    for sentence in english_data:
        sentence = sentence.lower()
        seg_list_exact = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(seg_list_exact)
        english_object_list.extend(
            [word for word, pos in tagged_words if pos.startswith("N")]
        )
    object_list = chinese_object_list + english_object_list
    return object_list


def stopword_filtering(seg_list_exact):
    global stopwords
    # Ensure stopwords is a set for faster lookups
    stopwords = set(stopwords)
    # Use a list comprehension to filter out stopwords
    object_list = [
        word
        for word in seg_list_exact
        if word not in stopwords and word not in {" ", "\xa0"}
    ]
    return object_list


def text2WordList(text):
    # remove all puncuation mark and emoji in text
    string_data = text_preprocessing(text)
    # seperate all content as word segments
    object_list = text_segmentation(string_data)
    #  filter meaningless words
    filtered_object_list = stopword_filtering(object_list)
    return filtered_object_list


def content_sentiment(wordlist):
    sentimentScore_list = []
    for word in wordlist:
        sentimentScore_list.append(SnowNLP(word).sentiments)
    average = sum(sentimentScore_list) / len(sentimentScore_list)
    return average


def jsonltojson(folder_path):
    global yearMismatchCount
    global JSONDecodeErrorCount
    # Dictionary to hold unique data
    _id_set = set()
    data = {}
    for file_name in glob.glob(folder_path + "/*.jsonl"):
        print(file_name)
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            num_lines = len(lines)
            file_year = file_name.split("_")[3]
            for counter, line in enumerate(lines, start=1):
                try:
                    json_line = json.loads(line)
                    _id = json_line.get("_id")
                    tweet_year = json_line.get("created_at").split("-")[0]
                    if file_year != tweet_year:
                        break
                    if _id is not None and _id not in _id_set:
                        _id_set.add(_id)
                        tweet_content = json_line.get("content")
                        wordlist = text2WordList(tweet_content)
                        json_line["content_wordlist"] = wordlist
                        json_line["content_sentiment"] = SnowNLP(
                            tweet_content
                        ).sentiments
                        data[_id] = json_line
                    else:
                        print(f"Skipping line due to yearMismatchError: {line}")
                        yearMismatchCount += 1
                except json.JSONDecodeError:
                    print(f"Skipping line due to JSONDecodeError: {line}")
                    JSONDecodeErrorCount += 1
                print(round(counter / num_lines, 1))

    # Convert the dictionary values to a list
    data = list(data.values())

    if not os.path.exists(
        "processed_data"
    ):  # Checks if the directory 'processed_data' exists
        os.mkdir("processed_data")  # Creates a new directory named 'processed_data'

    # Write the data without duplicates back to the JSON file
    with open(
        "processed_data/" + folder_path.split("\\")[1] + ".json", "w", encoding="utf-8"
    ) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)  # Use indent parameter


stopwords = create_stopword()
for folder in os.listdir("data"):
    jsonltojson(os.path.join("data", folder))
print(
    "yearMismatchCount: ",
    yearMismatchCount,
    "\nJSONDecodeErrorCount: ",
    JSONDecodeErrorCount,
)
