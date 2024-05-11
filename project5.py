import os
import pandas as pd
from tinytag import TinyTag
import librosa
import numpy as np
from pymongo import MongoClient
import streamlit as st

# Connect to MongoDB
mongo_uri = "mongodb://localhost:27017"
client = MongoClient(mongo_uri)

# Select/Create database
db = client['features_mfcc']

# Select/Create collection (similar to a table)
collection = db['spotify']

# Check if the collection is empty
if collection.count_documents({}) == 0:
    # Directory containing the audio files
    audio_dir = 'sample/001/'

    # List to store the data
    data = []

    # Iterate over all files in the directory
    for filename in os.listdir(audio_dir):
        if filename.endswith('.mp3'):
            # Full path to the audio file
            filepath = os.path.join(audio_dir, filename)

            # Extract metadata using tinytag
            tag = TinyTag.get(filepath)
            title = tag.title
            artist = tag.artist
            album = tag.album

            # Load the audio file and calculate MFCC
            y, sr = librosa.load(filepath)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)

            # Average the MFCC over time
            mfcc_avg = np.mean(mfcc, axis=1)

            # Append the data to the list
            data.append([filename,title, artist, album, mfcc_avg.tolist()])

    # Convert the list to a list of dictionaries and insert it into the MongoDB collection
    data = pd.DataFrame(data, columns=['filename','Title', 'Artist', 'Album', 'MFCC']).to_dict('records')
    collection.insert_many(data)

    print(f'Successfully inserted {len(data)} documents into the MongoDB collection.')
else:
    # Load the data from MongoDB
    data = list(collection.find({}))

    for document in data:
        print(document)
    print(f'Successfully loaded {len(data)} documents from the MongoDB collection.')


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

# Load the data from MongoDB
data = list(collection.find({}))

# Extract the MFCC features and convert them to a 2D numpy array
mfcc_features = np.array([doc['MFCC'] for doc in data])

if os.path.exists('model.joblib'):
    # Load the model from the 'model.joblib' file
    model = load('model.joblib')
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
    # Fit the NearestNeighbors model to the MFCC features
    model = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(mfcc_features)
    # Save the model to a 'model.joblib' file
    dump(model, 'model.joblib')


# To make a recommendation for a new user, select a random song
if not os.path.exists('history.txt'):
    with open('history.txt', 'w') as f:
        pass

with open('history.txt', 'r') as f:
    history = f.read().splitlines()




# Display the history in the sidebar
st.sidebar.title('History')
for song in history:
    st.sidebar.text(song)

# Streamlit app
st.title('PotyFy')
selected_song_index = np.random.choice(len(data))
with open('number.json', 'r') as f:
    selected_song_index = json.load(f)
selected_song_features = data[selected_song_index]['MFCC']

# Find the 5 songs that have the most similar MFCC features
distances, indices = model.kneighbors([selected_song_features])

# Display the selected song and provide a link to play it
st.write(f"Selected Song: {data[selected_song_index]['Title']}")

st.audio(os.path.join('sample/001/', data[selected_song_index]['filename']))

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

# Display the recommended songs
st.write("Recommended Songs:")
for no,i in enumerate(indices[0]):
    if no!=0:
        if st.button(f'Play {data[i]["Title"]}'):
            with open('number.json', 'w') as f:
                json.dump(int(i), f)
            with open('history.txt', 'a') as f2:
                f2.write(f'{data[i]["Title"]} by ({data[i]["Artist"]})\n')

