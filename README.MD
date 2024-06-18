# Stage 1

Implement any NLP model for this problem: https://www.kaggle.com/c/commonlitreadabilityprize/overview  
The final RMSE metric doesn't matter here, the goal here is just to get a trained model.    
Output:
- python script for training
- trained model with metrics
- README with information on how to reproduce our solution

## Solution

### Files

- `data/train.csv`: The dataset
- `script/train_model.py`: Python script to train the regression model
- `model/trained_model.pkl`: Trained model
- `model/metrics.txt`: Metrics of the trained model
- `README.md`: README with information on how to reproduce the solution

### Steps to Reproduce in Google Colab

1. Go to https://colab.research.google.com/
2. Create a new Notebook
3. Paste the script from the `script/train_model.py` file to the Notebook
4. Run the block of code
5. Colab will ask you to upload the file. Upload `data/train.csv` file
6. After execution will be finished download the model `trained_model.pkl` from the colab `content` folder

# Stage 2

Write an API server for the trained model form Stage1. The API should contain only 1 endpoint for prediction:    
Output:
- python script for running API
- README with information how to reproduce our solution

## Solution

### Files

- `api/api_server.py`: Python script for running API

### Libs

The next libs have to be installed:

- fastapi
- uvicorn
- pandas
- scikit-learn
- joblib

### How to run the server

1. Go to the repository root folder
2. Run `uvicorn api.api_server:app --host localhost --port 8080`

### How to test the server

Execute the next curl command in the CLI:  
`curl -X POST "http://127.0.0.1:8080/predict" -H "Content-Type: application/json" -d "{\"text\": \"Sample text to predict\"}"`

For the Linux-based OS may be necessary to change brackets: `curl -X POST "http://127.0.0.1:8080/predict" -H "Content-Type: application/json" -d '{"text": "Sample text to predict"}'`

# Stage 3

The solution was deployed to Google Cloud.  
Execute the next curl command in the CLI to check it:

`curl -X POST "https://ml-7gh53poy7a-ey.a.run.app/predict" -H "Content-Type: application/json" -d "{\"text\": \"Sample text to predict\"}"`

For the Linux-based OS may be necessary to change brackets: `curl -X POST "https://ml-7gh53poy7a-ey.a.run.app/predict" -H "Content-Type: application/json" -d '{"text": "Sample text to predict"}'`