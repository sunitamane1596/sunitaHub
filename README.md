# sunitaHub

## Predictive maintenance RUL web app

This Flask app estimates the remaining useful life (RUL) of equipment from its
operating measurements. It reads the training data from
`data/RUL_prediction_dataset.csv` and serves a form at `http://127.0.0.1:5000/`.

### Run on Windows

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/`, edit the prefilled equipment fields, and select
**Predict remaining life**. The JSON prediction endpoint is `POST /predict`.
