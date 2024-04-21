from pymongo import MongoClient
from pymongo.server_api import ServerApi
import json

# Your MongoDB Atlas connection string
WeiboDataset1_CONNECTION_STR = 'mongodb+srv://:@weibodataset1.3fbrcoi.mongodb.net/?retryWrites=true&w=majority'
WeiboDataset2_CONNECTION_STR = 'mongodb+srv://:@weibodataset2.bvjbm0q.mongodb.net/?retryWrites=true&w=majority'
def insert_documents(db, collection_name, topic):
    try:
        # WeiboDataset1_client = MongoClient(WeiboDataset2_CONNECTION_STR, server_api=ServerApi('1'))
        Database_client = MongoClient(WeiboDataset1_CONNECTION_STR, server_api=ServerApi('1'))  # 30-second timeout
        db = Database_client[db]
        collection = db[collection_name]

        # Read data from the JSON file
        with open(f'processed_data/{topic}.json', 'r', encoding='utf-8') as file:
            data = json.load(file)

        batch_size = 10000  # Adjust batch size as needed
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            result = collection.insert_many(batch)
            print(f"Inserted {len(result.inserted_ids)} documents")

    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        Database_client.close()

# Example usage
if __name__ == "__main__":
    insert_documents("weibodataset1", r"girls help girls(hashtag)", r'%23girls%20help%20girls%23')
