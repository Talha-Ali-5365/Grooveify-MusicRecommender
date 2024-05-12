import os
import pandas as pd
import numpy as np
import json
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from joblib import dump, load
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors

# Connect to MongoDB
mongo_uri = "mongodb://localhost:27017"
spark = SparkSession.builder.appName("music_recommender").config("spark.mongodb.input.uri", mongo_uri).config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1").getOrCreate()

# Load data from MongoDB
data = spark.read.format("mongo").option("uri", mongo_uri + "/features_mfcc.spotify").load()
data = data.drop('_id', 'label', 'label_num', 'text')
data = data.withColumnRenamed('Artist', 'label')

# Convert the 'MFCC' column to a vector
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
data = data.withColumn('features', list_to_vector_udf('MFCC'))

# Index the label column 'Artist'
labelIndexer = StringIndexer(inputCol='label', outputCol='indexedLabel').fit(data)
data = labelIndexer.transform(data)

# Split the data into training and test sets
train, test = data.randomSplit([0.8, 0.2], seed=12345)

# Define the ANN architecture
input_layer_size = len(data.select('MFCC').first()[0])
output_layer_size = data.select('indexedLabel').distinct().count()
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

# Load the MFCC features from MongoDB
mfcc_features = np.array(data.select('MFCC').collect())

# Fit the NearestNeighbors model to the MFCC features
model = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(mfcc_features)

# Save the model to a 'model.joblib' file
dump(model, 'model.joblib')

# Streamlit app
st.title('PotyFy')

# Load the history
with open('history.txt', 'r') as f:
    history = f.read().splitlines()

# Display the history in the sidebar
st.sidebar.title('History')
for song in history:
    st.sidebar.text(song)

# Recommend a song
selected_song_index = np.random.choice(len(data))
with open('number.json', 'r') as f:
    selected_song_index = json.load(f)
selected_song_features = data.select('MFCC').collect()[selected_song_index]

# Find the 5 songs that have the most similar MFCC features
distances, indices = model.kneighbors([selected_song_features])

# Display the selected song and provide a link to play it
st.write(f"Selected Song: {data.select('Title').collect()[selected_song_index]}")
st.audio(os.path.join('sample/001/', data.select('filename').collect()[selected_song_index]))

# Display the recommended songs
st.write("Recommended Songs:")
for i, index in enumerate(indices[0]):
    if i != 0 and st.button(f'Play {data.select("Title").collect()[index]}'):
        with open('number.json', 'w') as f:
            json.dump(int(index), f)
        with open('history.txt', 'a') as f2:
            f2.write(f'{data.select("Title").collect()[index]} by ({data.select("Artist").collect()[index]})\n')
