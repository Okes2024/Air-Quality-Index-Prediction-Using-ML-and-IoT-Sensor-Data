Air Quality Index Prediction Using ML and IoT Sensor Data

This project demonstrates how **synthetic IoT sensor data** can be used to build machine learning models for **Air Quality Index (AQI) prediction**.  
The workflow covers synthetic dataset generation, AQI computation, feature engineering, model training, evaluation, and visualization.

---

## 📂 Project Structure
.
├── air_quality_iot_ml.py # Main script (data generation + ML models)
├── synthetic_air_quality_dataset.xlsx # Example synthetic dataset
├── synthetic_air_quality_iot.csv # Dataset (CSV format)
├── residuals_plot.png # Residuals visualization
├── model_feature_importance.png # Feature importance (if tree-based model selected)
├── trained_aqi_model.joblib # Serialized trained model
└── README.md # Project documentation

yaml
Copy code

---

## 🚀 Features
- Generates a **reasonably populated synthetic dataset** simulating IoT air-quality sensors.  
- Includes **timestamps, sensor IDs, geolocation, meteorological parameters, pollutant levels (PM2.5, PM10, etc.), and computed AQI**.  
- Implements **machine learning models** (Ridge Regression, Random Forest, Gradient Boosting) to predict AQI.  
- Provides **cross-validation, performance metrics (RMSE, MAE, R²)**, and visualizations.  
- Saves dataset in **CSV/Excel** and trained model in `.joblib` format.

---

## 🛠️ Requirements
Install dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl joblib
▶️ Usage
Clone this repository:

bash
Copy code
git clone https://github.com/username/air-quality-ml-iot.git
cd air-quality-ml-iot
Run the main script:

bash
Copy code
python air_quality_iot_ml.py
Check outputs:

synthetic_air_quality_iot.csv / .xlsx → dataset

residuals_plot.png → residual analysis

model_feature_importance.png → feature contribution (if applicable)

trained_aqi_model.joblib → saved ML model

📊 Example Outputs
Residuals Plot: shows prediction errors vs. predicted AQI

Feature Importance: highlights key drivers (pollutants, meteorology, traffic)

Evaluation Metrics: RMSE, MAE, R²

📖 Background
Air pollution poses significant threats to public health, climate, and urban sustainability.
By leveraging IoT-enabled sensors and machine learning models, this project shows how AQI can be monitored and predicted in near real-time, helping cities and researchers design better air quality management strategies.

👨‍💻 Author
Okes Imoni
Data Scientist | Geospatial & Environmental Analytics | AI Researcher

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

⭐ Acknowledgements
Inspired by research in IoT-based air quality monitoring systems.

Uses open-source libraries: NumPy, Pandas, Scikit-learn, Matplotlib, Joblib.
