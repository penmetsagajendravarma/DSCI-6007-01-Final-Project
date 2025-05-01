# 📈 Time-Series Forecaster

*A zero-boilerplate Streamlit app that turns raw time-series data into tomorrow’s numbers—no PhD, no tangled notebooks, just click-and-go.*

---

## ✨ Why you’ll love it
- **Drag-n-drop CSV** → the app guesses dates, fills gaps, and finds the right cadence.
- **Model buffet** – pick classic ARIMA/SARIMA for transparency or LSTM for extra muscle.
- **Live feedback** – accuracy metrics and residual plots update as you tweak knobs.
- **Runs anywhere** – laptop-friendly, Docker-ready, AWS-approved.

---

## 🚀 Quick start

```bash
# 1. Grab the code
git clone https://github.com/penmetsagajendravarma/DSCI-6007-01-Final-Project.git
cd your-repo

# 2. Fire up a virtual env (optional but tidy)
python -m venv venv && source venv/bin/activate

# 3. Install what we need
pip install -r requirements.txt

# 4. Go!
streamlit run app.py

---

## 🗺️ Project tour (a.k.a. “What are all these files?”)

| File / folder | What’s inside |
|---------------|---------------|
| **app.py** | The Streamlit front-end—run this and you’re live. |
| **analysis.html** | Exploratory Data Analysis in a single self-contained HTML (open in any browser). |
| **forecast.html** | Interactive ARIMA vs LSTM comparison—see which model wins on your data. |
| **error.html** | Sample error page we surface when the upload or model goes sideways. |
| **result.html** | Clean overlay of *actual vs predicted* values, ready for screenshots. |
| **Tesla-Dataset.csv** | A tidy TSLA time-series you can play with right away. |
| **Arima-Model.png** | A snack-sized diagram of the ARIMA workflow—handy for reports. |
| **Section-01_Team-02_Final_Project-Technical_Report.docx** | The 10-page deep dive: methods, results, references. |
| **DS6007-01-Section01-Team02-Final project.pptx** | A crisp 12-slide deck—use it for lightning talks or demo day. |

*(Everything else—`requirements.txt`, Dockerfile, etc.—lives in the usual places.)*

---

## 📊 Sample run in 30 seconds

1. Fire up the app with  
   ```bash
   streamlit run app.py



