# Create Your Own Spotify Experience

## Project Overview
This project aims to create a Spotify-like music streaming service. It will feature a machine learning-powered music recommendation system, playback and streaming capabilities, and real-time suggestions based on user activity. The project will utilize the Free Music Archive dataset, Apache Spark for model training, and Apache Kafka for real-time recommendations. The final product will be a user-friendly web application.

## Team Members

`Talha Ali (22i-1971)`

`Shahzer Nadeem (22i-2043)`

## Dependencies

- **pandas**: For data manipulation and analysis.
- **streamlit**: To create the web application.
- **librosa**: For music and audio analysis.
- **KafkaConsumer**: From kafka-python, used for real-time data streaming.
- **MongoClient**: From PyMongo, used to work with MongoDB.
- **MultilayerPerceptronClassifier**: From PySpark MLlib, used for model training.
- **SparkSession**: From PySpark SQL, used for creating a session to use Spark.

## Usage

This project consists of two main Python files: `producer.py` and `project.py`.

1. **Running the Scripts**: To start the application, you need to run both `producer.py` and `project.py` at the same time. This can be done by opening two separate command line interfaces (CLI), navigating to the project directory in each, and running each script with a Python command.

2. **Accessing the Web Application**: Once both scripts are running, you can open your web browser and navigate to the Streamlit app to start interacting with the music streaming service. The Streamlit app will provide a user-friendly interface where you can listen to songs and receive personalized song recommendations based on your activity.


## Phase 1: ETL Pipeline
### Dataset
The `fma_large.zip` dataset is a comprehensive collection of 106,574 tracks, each lasting 30 seconds, spanning 161 genres. The dataset, when compressed, totals 93 GiB in size. It's ideal for music information retrieval and recommendation systems.

### Feature Extraction
In this project, Mel-Frequency Cepstral Coefficients (MFCC) was used to convert the audio tracks from the `fma_large.zip` dataset into numerical formats. This process helped in capturing the audio's characteristics, which were then used to train the music recommendation model.

### Data Storage
In this project, `MongoDB`, a NoSQL database, was used to store the transformed audio features. After the feature extraction process using MFCC, the numerical data was loaded into MongoDB. This provided a scalable and efficient solution for handling the large dataset, allowing for easy retrieval of data during the model training phase.

## Phase 2: Music Recommendation Model
### Model Training
In this project, the music recommendation model was trained using `Apache Spark`, a powerful tool for handling big data. The algorithm employed was Approximate Nearest Neighbors (ANN), which is effective for recommendation systems. `ANN` works by finding the 'nearest' items to a given item in a multi-dimensional space, which in this case are the audio features of the tracks. The 'closeness' of items is determined based on their similarity in features. The model was trained on the numerical data obtained from the MFCC feature extraction, which was stored in MongoDB. The trained model can then predict recommendations based on the 'closeness' of tracks in the feature space.

### Model Evaluation
The primary evaluation metric used to assess the model in this project was accuracy. After training the Approximate Nearest Neighbors (ANN) model using Apache Spark, the model's predictions were compared with the actual values to calculate the accuracy. Following hyperparameter tuning, the model achieved an accuracy of **40%**. This indicates that the model correctly predicted the recommendations 40% of the time. While this is a good start, further optimization and tuning could potentially improve this accuracy.

## Phase 3: Deployment
### Web Application
The interactive music streaming web application was developed using `Streamlit`, a popular open-source framework for creating data apps. Streamlit allows for rapid prototyping and the ability to integrate machine learning models, making it an ideal choice for this project. The web application incorporates the trained music recommendation model and provides real-time music recommendations. It offers a user-friendly interface where users can interact, play music, and receive personalized song suggestions based on their activity.

### Real-Time Recommendations
In this project, `Apache Kafka`, a real-time data streaming platform, was leveraged to generate music recommendations based on user activity. The system was designed with a producer-consumer model. The producer loads data from MongoDB, which contains the transformed audio features. This data is then sent to the consumer in chunks. The consumer keeps appending the incoming data, thereby maintaining an up-to-date dataset. Based on this real-time and updated data, the consumer generates unique music recommendations for each user. This ensures that the recommendations stay relevant and personalized, enhancing the user's music streaming experience.

### Findings and Results
The developed music recommendation system successfully provided personalized song suggestions based on user activity. The use of Apache Kafka for real-time data streaming and ANN model trained using Apache Spark were effective. The system achieved an accuracy of 40%, indicating room for further optimization. Overall, the project demonstrated the potential of machine learning in enhancing music streaming services.

## Conclusion
The project successfully developed a music streaming service with a personalized recommendation system. It demonstrated the effective use of Apache Kafka, Apache Spark, and ANN model. However, with an accuracy of 40%, there's room for improvement. Future work could explore different algorithms, feature extraction methods, or incorporate user feedback to enhance the system's performance and user experience.

## References
https://docs.streamlit.io/develop/api-reference
https://spark.apache.org/docs/latest/api/python/index.html
https://scikit-learn.org/stable/modules/neighbors.html#:~:text=The%20label%20assigned%20to%20a,value%20specified%20by%20the%20user.

Chat-gpt is also used to some extent, for help.
