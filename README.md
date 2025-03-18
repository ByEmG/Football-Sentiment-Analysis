# Football-Sentiment-Analysis
⚽️ Beyond The Score – Football Sentiment Analysis

📌 Project Overview

This project analyzes public sentiment about popular football players and events using machine learning and natural language processing (NLP). It integrates data scraped from Reddit, Twitter, and popular sports websites like ESPN to derive insights beyond traditional match statistics.

🎯 Objectives

Understand public perception of football players related to match performance, transfers, injuries, and off-field behaviors.

Implement machine learning techniques to classify sentiment as positive, negative, or neutral.

Provide visualizations to clearly interpret sentiment analysis outcomes.

🛠️ Tools and Technologies

Python: Data collection, cleaning, modeling, and visualization

Pandas & NumPy: Data manipulation and analysis

BeautifulSoup & Requests: Web scraping from ESPN, Sky Sports, Reddit

Tweepy: Twitter API for fetching tweets

NLTK: Text preprocessing

Scikit-learn: TF-IDF vectorization, Logistic Regression, Random Forest

Matplotlib, Seaborn, WordCloud: Data visualization

📂 Project Structure

Football-Sentiment-Analysis/
├── data/
│   ├── final_sentiment_analysis.csv
│   └── final_cleaned_sentiment_analysis.csv
├── scripts/
│   ├── cleaning.py
│   ├── TD_IDF.py
│   ├── label_encoding.py
│   ├── train.py
│   ├── model_training.py
│   └── visualize.py
├── results/
│   ├── sentiment_distribution.png
│   ├── positive_wordcloud.png
│   ├── negative_wordcloud.png
│   └── neutral_wordcloud.png
└── README.md

🚀 How to Run the Project

Step 1: Clone the repository

git clone https://github.com/ByEmG/Football-Sentiment-Analysis.git
cd Football-Sentiment-Analysis

Step 2: Install dependencies

pip install -r requirements.txt

Step 3: Run scripts

Data Cleaning:

python scripts/cleaning.py

Feature Extraction & Modeling:

python scripts/model_training.py

Visualization:

python scripts/visualize.py

📈 Key Results

Achieved high accuracy in sentiment classification using Logistic Regression and Random Forest.

Visual insights clearly demonstrate relationships between public sentiment and real-world football outcomes.

📊 Visualizations

Sentiment Distribution:  

Word Clouds:

Positive Sentiment: 

Negative Sentiment: 

Neutral Sentiment: 

🔮 Future Improvements

Expand data sources and integrate real-time analytics.

Explore advanced NLP techniques (transformers like BERT or GPT).

Develop an interactive web application for real-time sentiment visualization.

📜 References

scikit-learn documentation

NLTK documentation

Pandas Documentation

Tutorials and articles from Medium and Kaggle.

🌟 Thank you for checking out my project!

Feel free to connect with me on LinkedIn for any feedback or collaborations!

