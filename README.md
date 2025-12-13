# BloomAI-FinalCode

# Project Structure
```
Fine-Tuned-Models
|
|-- roberta-large # Contains the model files for the RoBERTa Model
|-- distilbert-base-uncased # Contains the model files for the Distilbert Model

data
|
|-- blooms_taxonomy_dataset.csv # Dataset of questions labeled with Bloom levels

fine_tuning
|
|-- data_processing.py # Loads, and prepares the dataset
|-- testing_metrics.py # Evaluates the models performance and shows metrics
|-- training.py # Fine-tunes the two models

flaskApp
|
|-- static/css/main.css # Styling for the web interace
|-- templates 
    |-- base.html # HTML layout
    |-- index.html # User interface for predictions
|-- app.py # Flask backend 
|-- predict2.py # Loads trained model and generates predictions
```
