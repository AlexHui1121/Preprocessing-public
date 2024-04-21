from pymongo import MongoClient, UpdateOne
from pymongo.server_api import ServerApi
from datetime import datetime

WeiboDataset1_CONNECTION_STR = 'mongodb+srv://:@weibodataset1.3fbrcoi.mongodb.net/?retryWrites=true&w=majority'
WeiboDataset2_CONNECTION_STR = 'mongodb+srv://:@weibodataset2.bvjbm0q.mongodb.net/?retryWrites=true&w=majority'

client = MongoClient(WeiboDataset1_CONNECTION_STR, server_api=ServerApi('1'))
db = client['weibodataset1']
collection = db['girls help girls(hashtag)']

# Get all documents
documents = collection.find({})

updates = []

for doc in documents:
    # Convert the 'created_at' field to a datetime object
    created_at = datetime.strptime(doc['created_at'], '%Y-%m-%d %H:%M:%S')

    # Create an UpdateOne object
    update = UpdateOne({'_id': doc['_id']}, {'$set': {'created_at': created_at}})

    updates.append(update)

# Update the documents
collection.bulk_write(updates)
