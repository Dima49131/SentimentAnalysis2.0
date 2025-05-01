# Current Live Demo
https://dima333212.pythonanywhere.com/

# Sentiment Analysis 2.0
This is my second attempt and making a sentiment analysis tool. This time i've leveraged a larger amount of data to train the model on. Specifically using Amazon reviews from 2018, found here https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

I trained it on 2,205,054 total reviews from the data set.
735,018 positive, negative 735,018, and  735,018 neutral.
 
The model_files folder contains the full trained model, if you want to train your own delete the folder and it will build a new one in its place.


# Dependencies
```
pip install -r requirements.txt
```
# Running
```
python app.py
```
go to local host http://127.0.0.1:5000/



