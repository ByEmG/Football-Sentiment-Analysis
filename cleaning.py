import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("final_sentiment_analysis.csv")

# Replace NaN values clearly in the Text column
df['Text'] = df['Text'].fillna('')

# clearly defined text preprocessing function
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)           # remove URLs
    text = re.sub('[^a-zA-Z]', ' ', text).lower() # keep letters only
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply cleaning function clearly
df['cleaned_text'] = df['Text'].apply(preprocess_text)

# Save cleaned dataset clearly
df.to_csv("final_cleaned_sentiment_analysis.csv", index=False)

print("✅ Dataset cleaned successfully and saved as final_cleaned_sentiment_analysis.csv!")