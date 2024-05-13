import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pymongo import MongoClient
from confluent_kafka import Producer
import time



mongo_uri = "mongodb://localhost:27017"
client = MongoClient(mongo_uri)


# Select/Create database
db = client['features_mfcc']

# Select/Create collection (similar to a table)
collection = db['spotify']

result_all = collection.find()


KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
topic1 = "topic1"
topic2 = "topic2"
topic3 = "topic3"

kafka_producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")

print('Sending data to consumer...')
while True:
# Fetch data from MongoDB and publish to Kafka
    for document in result_all:
        kafka_producer.produce(topic1, str(document).encode('utf-8'), callback=delivery_report)
        kafka_producer.produce(topic2, str(document).encode('utf-8'), callback=delivery_report)
        kafka_producer.produce(topic3, str(document).encode('utf-8'), callback=delivery_report)
        time.sleep(0.1)
        #print(document)

    # Wait for all messages to be delivered
    kafka_producer.flush()
