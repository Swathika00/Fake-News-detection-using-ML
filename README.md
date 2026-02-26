# Fake News Detection System

A production-ready Fake News Detection web application using Machine Learning and NLP framework.

## Features

- **User Authentication**: Register, login, and logout functionality
- **Single Prediction**: Enter news text and get instant fake/real detection
- **Batch Predictions**: Upload CSV files for bulk analysis
- **Prediction History**: View all past predictions with details
- **REST API**: FastAPI endpoints for programmatic access

## Label Mapping

- **0 = Fake News**
- **1 = Real News**

## Model Details

- **Algorithm**: TF-IDF + Random Forest Classifier
- **Accuracy**: 90% test accuracy
- **Cross-Validation**: 84% mean accuracy

## Installation

1. **Install dependencies:**
```
pip install -r requirements.txt
```

2. **Train the model:**
```
python train_model.py
```

3. **Run the application:**
```
python app.py
```

## Running the Application

The Flask web application runs at: **http://127.0.0.1:5000**

### Routes

| Route | Description |
|-------|-------------|
| `/` | Home page (redirects to login) |
| `/register` | User registration |
| `/login` | User login |
| `/dashboard` | Main dashboard with predictions |
| `/batch` | Batch predictions from CSV |
| `/history` | View prediction history |
| `/logout` | User logout |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Single prediction |
| `/api/health` | GET | Health check |

## API Usage

### Single Prediction

```
bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Your news article text here"}'
```

Response:
```
json
{
    "prediction": 0,
    "confidence": 85.5,
    "label": "Fake News"
}
```

### Health Check

```
bash
curl http://127.0.0.1:5000/api/health
```

## Project Structure

```
.
├── app.py                  # Flask web application
├── api.py                  # FastAPI application
├── train_model.py          # Model training script
├── fake_news_pipeline.pkl  # Trained model
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/             # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── result.html
│   ├── batch.html
│   ├── batch_result.html
│   ├── history.html
│   └── error.html
├── uploads/               # Uploaded batch files
└── instance/              # SQLite database
```

## Requirements

- Python 3.8+
- Flask
- Scikit-learn
- Pandas
- NumPy
- Flask-SQLAlchemy

## License

This project is licensed under the MIT License.
