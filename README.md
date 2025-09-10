# Vibe-Coding-Ai-Model

---

## VibeCoder: Python Code Quality Analyzer

VibeCoder is an AI-powered Python code quality analyzer with an interactive Streamlit dashboard.
It extracts code metrics (complexity, readability, Halstead metrics, AST depth, sentiment, etc.), trains ML models, and predicts code quality levels (excellent, good, average, poor).

## The app can:

Analyze raw Python code you paste in
Train models on Python files pulled from GitHub repositories
Visualize feature distributions in a dashboard
Save/load trained models for reuse

## Features

Feature extraction: lines of code, comments, functions, imports, cyclomatic complexity, nesting depth, readability score, sentiment, Halstead metrics, AST depth, etc.
Machine learning: Random Forest, Gradient Boosting, Voting Ensemble (with soft voting).
Training: Automatically builds datasets from GitHub repos and assigns heuristic quality labels.
Dashboard UI: Clean dark-themed Streamlit interface with four sections:
Analyze: Paste Python code and predict its quality.
Train: Fetch Python files from GitHub, build a dataset, and train models.
Features: Explore distributions of extracted features with plots.

# About: Project info and notes.

## Installation

1. Clone this repository
git clone https://github.com/your-username/vibecoder.git
cd vibecoder

2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt


Create a requirements.txt file:

streamlit
pandas
numpy
plotly
scikit-learn
nltk
requests

4. Download NLTK data

VibeCoder needs some NLTK resources. They’ll auto-download on first run, but you can pre-install them:

import nltk
for r in ("punkt", "vader_lexicon", "stopwords"):
    nltk.download(r)

---

## Running the App

From inside the project folder:

streamlit run vibecoder_dashboard_app.py
Open your browser at http://localhost:8501

---

## Usage

Analyze Code

Go to the Analyze tab.
Paste Python code in the text area.
Click Predict quality to see predictions, probabilities, and feature values.
Train a Model
Go to the Train tab.
Enter a GitHub repo URL (e.g., https://github.com/python/cpython](https://github.com/samyakrajbayar/Vibe-Coding-Ai-Model).
Click Fetch repo → this downloads .py files.
Click Train now → trains ML models on extracted features.
View accuracy and dataset samples.

## Feature Dashboard

Go to Features after training.
See boxplots for key metrics (complexity, readability, sentiment, etc.).
Explore histograms for individual features.

## Save & Load Models

Use the sidebar buttons Save and Load to persist trained models to .pkl files.

---

## Project Structure

vibecoder/
│
├── vibecoder_dashboard_app.py   # Main Streamlit app
├── requirements.txt             # Dependencies
├── code_dataset/                # (Created dynamically for repo pulls)
└── vibecoder.pkl                # (Optional saved model file)


## Quality labels (excellent, good, average, poor) are heuristic. You may replace them with real data for serious use.

GitHub API rate limits may apply if fetching large repos.

UI styling is experimental (dark theme with custom CSS).
