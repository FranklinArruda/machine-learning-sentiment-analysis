# Income Prediction and Sentiment Analysis – Final Assignment (*Machine Learning*)

![machine](https://github.com/user-attachments/assets/ef18e1fc-7438-49ef-a437-474be9afd81d)


This repository contains the final assignment project for the module on Machine Learning and Natural Language Processing. The project is divided into two main parts:

- **Part 1**: Income Prediction using supervised machine learning.
- **Part 2**: Sentiment Analysis on social media text data using Natural Language Processing techniques.

---

## Part 1: Income Prediction (Regression Task)

### Objective
To predict a customer's annual income based on structured features such as age, education, and work class.

### Approach
- Cleaned and preprocessed the dataset (handled missing values, encoded categorical variables).
- Trained and compared two regression models:
  - Linear Regression
  - Random Forest Regressor
- Evaluated model performance using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score

### Outcome
- Random Forest performed better in all metrics.
- Final model selected: **Random Forest Regressor**
- Used the model to predict the income of a new customer: **~£54,030 per year**

---

## Part 2: Sentiment Analysis on Twitter Data

### Objective
To classify the sentiment of real-world tweets as Positive, Negative, or Neutral using Natural Language Processing techniques.

### Approach
- Sampled 10,000 tweets from a dataset containing 1.6 million entries.
- Preprocessed the tweets:
  - Removed punctuation, stopwords, and mentions
  - Normalized and cleaned the text
- Used **VADER Sentiment Analyzer** (from NLTK) for sentiment classification.
- Mapped compound scores to sentiment labels.
- Visualized sentiment distribution with bar charts.
- Compared predicted sentiments with original labels using a confusion matrix.

---

## Additional Features

- Well-structured notebook with clearly written markdown explanations for each step.
- Inline references to external sources used for guidance and troubleshooting.
- Embedded evaluation images and a custom illustration representing personal reflection.
- Transparent version control with descriptive commit messages tied to specific learning steps.

---

## How to Run

1. Clone or download this repository.
2. Open the Jupyter Notebook (.ipynb file).
3. Install the required libraries if not already available:
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `scikit-learn`
   - `nltk`
4. Run the following lines once to download necessary NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('vader_lexicon')

## Personal Reflection
This project was an opportunity to apply everything I’ve learned throughout the module. I made a conscious effort to apply the feedback I received from previous assignments, particularly around vague explanations. This time, I structured my notebook more clearly, added meaningful markdown commentary, and explained why I made certain decisions.

I encountered several challenges, especially with dataset loading and text preprocessing, but managed to solve them through self-directed learning and reference to official documentation and tutorials. I also included these sources in my notebook for transparency.

In the end, this wasn’t just an assignment to submit, it felt like building something meaningful. I now feel much more confident applying machine learning and sentiment analysis in real-world contexts.

## References
- Sentiment140 Dataset – Kaggle
https://www.kaggle.com/datasets/kazanova/sentiment140

- DataCamp Tutorial – Sentiment Analysis with NLTK
https://www.datacamp.com/tutorial/text-classification-python

- SaturnCloud – Fixing UnicodeDecodeError in Pandas
https://saturncloud.io/blog/how-to-fix-the-pandas-unicodedecodeerror

- NLTK VADER Sentiment Documentation
https://www.nltk.org/api/nltk.sentiment.vader.html


## **License**

This project was developed as part of the coursework for the Data Analytics program at **CCT College Dublin**.

All code, explanations, and outputs in this project are submitted solely for educational purposes and assessment under the supervision of CCT College.

Unauthorized use or distribution is not permitted without written permission from the author or the institution.

© 2025 CCT College Dublin – All Rights Reserved.
