# Sentiment Analysis 2.0

A more advanced sentiment analysis tool trained on a large Amazon reviews dataset.

## About This Project

This is my second attempt at creating a sentiment analysis tool. This version leverages a significantly larger dataset for training - specifically Amazon reviews from 2018.

**Dataset Source**:  
[https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)

### Training Data
- **Total reviews**: 2,205,054
- **Positive reviews**: 735,018
- **Negative reviews**: 735,018
- **Neutral reviews**: 735,018

### Model Files
The `model_files` folder contains the full trained model. To train your own model:
1. Delete the existing `model_files` folder
2. Run the training process (a new folder will be created automatically)
3. Make sure you update app.py to have the json file path you download to `PREDEFINED_JSON_PATH`
## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Running
```
python app.py
```
go to local host http://127.0.0.1:5000/

### Side Note
Make sure the .txt file uploaded has each review on a seperate line to properly parse the file.

# Live Demo 
### https://dima333212.pythonanywhere.com/

<img width="1440" alt="Screen Shot 2025-05-01 at 1 10 18 AM" src="https://github.com/user-attachments/assets/61cc998d-58e5-464c-9456-1486e6ed81d1" />


