import os
import pandas as pd
from tinytag import TinyTag
import librosa
import numpy as np
import streamlit as st
from kafka import KafkaConsumer
import json
import subprocess
import ast
from pymongo import MongoClient
from sklearn.neighbors import NearestNeighbors
from joblib import dump, load
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
import json
import base64
from pathlib import Path

mongo_uri = "mongodb://localhost:27017"
client = MongoClient(mongo_uri)


# Select/Create database
db = client['features_mfcc']

# Select/Create collection (similar to a table)
collection = db['spotify']


bootstrap_servers = ['localhost:9092']
topic1 = 'topic1'
topic2 = 'topic2'
topic3 = 'topic3'

consumer1 = KafkaConsumer(topic1, bootstrap_servers=bootstrap_servers)
consumer2 = KafkaConsumer(topic2, bootstrap_servers=bootstrap_servers)
consumer3 = KafkaConsumer(topic3, bootstrap_servers=bootstrap_servers)

data = []

if 'comp' not in st.session_state:
    st.session_state.comp = 10
if 'temp_dic' not in st.session_state:
    st.session_state.temp_dic = {}
if 'song_index' not in st.session_state:
    st.session_state.song_index = 65

print('dic:')
print(st.session_state.temp_dic)

st.session_state.temp_dic = {}


temp=1
dat = []
print('Recieving data from consumers...')
while True:
    if temp%st.session_state.comp==0:
        break
    consumer_messages1 = consumer1.poll(timeout_ms=100)
    consumer_messages2 = consumer2.poll(timeout_ms=100)
    consumer_messages3 = consumer3.poll(timeout_ms=100)
    for message in consumer_messages1.values():
        for m in message:
            if (ast.literal_eval('{'+(m.value.decode('utf-8').split('), ')[1])+'\n')) not in data:
                dat.append(ast.literal_eval('{'+(m.value.decode('utf-8').split('), ')[1])+'\n'))
            #print(ast.literal_eval('{'+(m.value.decode('utf-8').split('), ')[1])+'\n'))
    for message in consumer_messages2.values():
        for m in message:
            if (ast.literal_eval('{'+(m.value.decode('utf-8').split('), ')[1])+'\n')) not in dat:
                dat.append(ast.literal_eval('{'+(m.value.decode('utf-8').split('), ')[1])+'\n'))
            #print(ast.literal_eval('{'+(m.value.decode('utf-8').split('), ')[1])+'\n'))
    for message in consumer_messages3.values():
        for m in message:
            if (ast.literal_eval('{'+(m.value.decode('utf-8').split('), ')[1])+'\n')) not in dat:
                dat.append(ast.literal_eval('{'+(m.value.decode('utf-8').split('), ')[1])+'\n'))
            #print(ast.literal_eval('{'+(m.value.decode('utf-8').split('), ')[1])+'\n'))
    temp+=1

print('\n\n\n\n')
print(len(dat))
data = list(collection.find({}))
#print(f'Successfully loaded {len(data)} documents from the MongoDB collection.')
# Connect to MongoDB

print('Training Model...')

# Extract the MFCC features and convert them to a 2D numpy array
mfcc_features = np.array([doc['MFCC'] for doc in data])

if os.path.exists('model.joblib'):
    # Load the model from the 'model.joblib' file
    print('Already trained model found!')
    print('Loading Model...')
    model = load('model.joblib')
    print('Model loaded successfully!')
else:
    spark = SparkSession.builder \
    .appName("music_recommender") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/features_mfcc.spotify") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .getOrCreate()

    # Load the data from MongoDB
    data = spark.read.format("mongo").option("uri","mongodb://localhost:27017/features_mfcc.spotify").load()
    data = data.drop('Unnamed: 0','_id','label','label_num','text')
    data.show()
    # Convert the 'MFCC' column to a vector

    # Define a UDF to convert array<double> into DenseVector
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())

    # Apply the UDF to the 'MFCC' column
    data = data.withColumn('MFCC', list_to_vector_udf('MFCC'))
    assembler = VectorAssembler(inputCols=['MFCC'], outputCol='features')
    data = assembler.transform(data)

    # Index the label column 'Artist'
    labelIndexer = StringIndexer(inputCol='Artist', outputCol='label').fit(data)
    data = labelIndexer.transform(data)

    # Optionally, you can drop the 'filename' column if it's not used for training
    data = data.drop('filename')

    # Split the data into training and test sets
    train, test = data.randomSplit([0.8, 0.2], seed=12345)

    # Define the ANN architecture
    # The number of nodes in the input layer should match the size of the MFCC feature vector
    # The number of nodes in the output layer should match the number of distinct artists
    input_layer_size = len(data.select('MFCC').first()[0])
    output_layer_size = data.select('label').distinct().count()
    layers = [input_layer_size, 128, 64, output_layer_size]

    # Create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

    # Train the model
    model = trainer.fit(train)

    # Evaluate the model on the test data
    result = model.transform(test)
    evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
    accuracy = evaluator.evaluate(result)
    print(f"Test set accuracy = {accuracy}")

    # Stop the Spark session
    spark.stop()
    #Fit the NearestNeighbors model to the MFCC features
    model = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(mfcc_features)
    # Save the model to a 'model.joblib' file
    dump(model, 'model.joblib')
    print('Training Completed Successfully!')

# To make a recommendation for a new user, select a random song
if not os.path.exists('history.txt'):
    with open('history.txt', 'w') as f:
        pass

with open('history.txt', 'r') as f:
    history = f.read().splitlines()

st.session_state.comp=80

def track(i):
    with open('number.json', 'w') as f:
        json.dump(int(i), f)
    with open('history.txt', 'a') as f:
        f.write(f'{data[i]["Title"]} by {data[i]["Artist"]}\n')

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# Display the history in the sidebar
st.sidebar.title(' ')
st.sidebar.markdown(f"## <span style='color: #f0463a;'>History</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")
for song in history:
    st.sidebar.text(song)

# Streamlit app
st.markdown("<h1 style='text-align: center; color: #f0463a;'>Groovefy</h1>", unsafe_allow_html=True)

selected_song_index = np.random.choice(len(data))
if os.path.exists('number.json'):
    with open('number.json', 'r') as f:
        selected_song_index = json.load(f)
selected_song_features = data[selected_song_index]['MFCC']

# Find the 5 songs that have the most similar MFCC features
distances, indices = model.kneighbors([selected_song_features])

# Display the selected song and provide a link to play it
st.markdown(
    "<img src='data:image/png;base64,{}' style='display: block; margin-left: auto; margin-right: auto;'>".format(img_to_bytes('img9.jpg')),
    unsafe_allow_html=True
)

st.markdown(f"## <span style='color: white;'>Selected Song: {data[selected_song_index]['Title']}</span>", unsafe_allow_html=True)

st.audio(os.path.join('sample/007/', data[selected_song_index]['filename']))

css = """
<style>
    .stButton>button {
        width: 100%;
        border: none;
    }
</style>
"""

# Apply the CSS to the app
st.markdown(css, unsafe_allow_html=True)

# Add a separator
st.markdown("---")

# Display the recommended songs
st.markdown("## <span style='color: white;'>Recommended Songs:</span>", unsafe_allow_html=True)
for no,i in enumerate(indices[0]):
    if no!=0:
        st.button(f'Play {data[i]["Title"]}',on_click=track,args=(i,))

            

