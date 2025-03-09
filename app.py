from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

# Load data
df = pd.read_csv("final_sentiment_analysis.csv")

@app.route("/")
def index():
    return render_template("index.html", data=df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)