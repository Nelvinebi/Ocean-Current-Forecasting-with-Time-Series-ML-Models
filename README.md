# 🌊 Ocean Current Forecasting with Time Series ML Models

This project simulates and forecasts ocean current speeds using synthetic time series data and machine learning models. It demonstrates how lag-based features and supervised ML can be used for environmental time series forecasting.

---

## 📌 Project Overview

Ocean current forecasting is crucial for marine navigation, climate research, and coastal management. This project generates synthetic weekly ocean current speed data and applies machine learning models—Linear Regression and Random Forest—to predict future current speeds based on past observations.

---

## 🧪 Technologies Used

- Python 3
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn

---

## 📁 Files

```bash
.
├── synthetic_ocean_current.csv    # Generated dataset (150 weekly samples)
├── ocean_current_forecasting.py   # Python script for analysis and forecasting
└── README.md                      # Project overview and usage instructions
📈 Features
✅ Synthetic data simulation with seasonality and trend

🔁 Lag-based time series to supervised transformation

🧠 ML models: Linear Regression and Random Forest

📉 Model evaluation using RMSE

📊 Visualization of forecast vs. actual values

⚙️ How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/ocean-current-forecasting.git
cd ocean-current-forecasting
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the script:

bash
Copy
Edit
python ocean_current_forecasting.py
📂 Dataset Preview
Date	Current_Speed (m/s)
2015-01-04	1.62
2015-01-11	1.48
...	...

📜 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Agbozu Ebingiye Nelvin
📧 nelvinebingiye@gmail.com
🔗 @nelvinebi
