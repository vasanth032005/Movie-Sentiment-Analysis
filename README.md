# 🎬 Movie Sentiment Analysis

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue?logo=pandas)
![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

</p>

---

# 📖 Project Overview

Movie Sentiment Analysis is a Natural Language Processing (NLP) project that predicts whether a movie review expresses a **Positive** or **Negative** sentiment.

The project uses traditional NLP techniques such as:

- Text Cleaning
- Tokenization
- Stopword Removal
- TF-IDF Vectorization
- Machine Learning Classification

A simple and interactive **Streamlit Web Application** allows users to enter a movie review and instantly receive a sentiment prediction.

---

# 🚀 Demo

### Positive Review

Input

```
This movie is excellent. The acting was amazing and the story was fantastic.
```

Prediction

```
😊 Positive Review
```

---

### Negative Review

Input

```
Worst movie ever. Waste of time.
```

Prediction

```
😞 Negative Review
```

---

# ✨ Features

✅ Text Preprocessing

- Lowercase Conversion
- Remove Special Characters
- Remove Punctuation
- Remove Numbers

✅ NLP

- Tokenization
- Stopword Removal
- TF-IDF Vectorization

✅ Machine Learning

- Logistic Regression

✅ Web Application

- Streamlit Interface
- Real-Time Prediction
- Emoji Output
- Prediction Confidence

---

# 📂 Project Structure

```
Movie-Sentiment-Analysis
│
├── dataset
│   └── IMDB_Dataset.csv
│
├── models
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── src
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│
├── images
│   ├── home.png
│   └── prediction.png
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

Dataset Used

**IMDb Movie Reviews Dataset**

- 50,000 Movie Reviews
- Binary Classification
- Positive Reviews
- Negative Reviews

Download

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

# ⚙️ Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Scikit-Learn | Machine Learning |
| Streamlit | Web Application |
| Joblib | Save Model |
| TF-IDF | Feature Engineering |

---

# 🧠 NLP Pipeline

```
Raw Text
      │
      ▼
Text Cleaning
      │
      ▼
Tokenization
      │
      ▼
Stop Words Removal
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Machine Learning Model
      │
      ▼
Prediction
```

---

# 📈 Machine Learning Workflow

```
Dataset

↓

Data Cleaning

↓

Text Preprocessing

↓

TF-IDF

↓

Train-Test Split

↓

Logistic Regression

↓

Model Evaluation

↓

Save Model

↓

Streamlit Deployment
```

---

# 💻 Installation

Clone Repository

```bash
git clone https://github.com/vasanth032005/movie-sentiment-analysis.git
```

Go to project

```bash
cd movie-sentiment-analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Train Model

```bash
python src/train_model.py
```

Run Streamlit

```bash
streamlit run app.py
```

Open browser

```
http://localhost:8501
```

---

# 📷 Screenshots

## Home Page

```
(Add Screenshot Here)
```

---

## Prediction

```
(Add Screenshot Here)
```

---

# 📌 Sample Reviews

## Positive

```
This movie was amazing.
```

Prediction

```
Positive 😊
```

---

## Negative

```
Terrible movie. Waste of time.
```

Prediction

```
Negative 😞
```

---

# 📊 Future Improvements

- Deep Learning (LSTM)
- BERT Model
- Transformer Model
- Attention Mechanism
- Movie Poster Integration
- Word Cloud
- Sentiment Score Visualization
- User Authentication
- Cloud Deployment

---

# 📚 Learning Outcomes

This project helped me understand:

- Natural Language Processing
- Text Cleaning
- Tokenization
- Stopword Removal
- TF-IDF
- Machine Learning
- Model Deployment
- Streamlit
- Git & GitHub

---

# 👨‍💻 Author

**S. Vasanth**

Artificial Intelligence & Data Science Student

GitHub

https://github.com/vasanth032005

LinkedIn

(Add LinkedIn Profile)

---

# ⭐ Support

If you found this project useful,

⭐ Star this repository

🍴 Fork this repository

📢 Share with others

---

# 📄 License

This project is licensed under the MIT License.

---

# 🙏 Acknowledgements

- Kaggle
- Scikit-Learn
- Streamlit
- Python Community
- IMDb Dataset

---

## ⭐ If you like this project, don't forget to give it a Star!
