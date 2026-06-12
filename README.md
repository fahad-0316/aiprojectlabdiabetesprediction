# Diabetes_Project

Simple Flask app using KNN to predict diabetes (Pima dataset).

How to run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py    # produces model.pkl and scaler.pkl
python app.py            # starts Flask app on http://127.0.0.1:5000
```

Files to note:
- `app.py` — Flask app
- `train_model.py` — trains model and saves `model.pkl` and `scaler.pkl`
- `templates/index.html` — UI
- `static/style.css` — styles

License: add one if needed
