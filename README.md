##Project Overview

This repository focuses on image classification, specifically determining whether a person in an image is happy or sad. The project includes:
-Data cleaning to ensure high-quality images.
-Preprocessing to scale pixel values between 0 and 1.
-Building a deep learning model using TensorFlow and Keras.
-Training the model on a dataset containing happy and sad images.
-Evaluating the model's performance using precision, recall, and binary accuracy metrics.
-Providing a Streamlit web application for real-time image classification.

##Project Structure
data/: Contains image datasets for happy and sad people.
models/: Holds the saved trained model (happysadmodel.h5).
image_classification.ipynb: Jupyter Notebook with the entire project pipeline.
streamlit_app.py: Streamlit web application for live image classification.

##Usage

*Clone the repository:
git clone https://github.com/your-username/ImageClassification.git
cd ImageClassification

*Install dependencies:
pip install -r requirements.txt

*Run the Streamlit app:
streamlit run app.py

Upload an image and see the model predict whether the person is happy or sad.

##Results
Trained a deep learning model achieving high accuracy on the validation set.
Deployed a Streamlit web application for real-time image classification.

Connect with Me
📧 Email: pablo.santilli7@gmail.com
🌐 LinkedIn: LinkedIn.com/in/pablosantilli

Feel free to explore, learn, and contribute to this exciting project! Your feedback is always welcome. 
