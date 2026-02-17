# 🚀 Startup Success Prediction using Machine Learning

This project leverages Machine Learning to predict whether a startup company will be **successful (Acquired)** or **unsuccessful (Closed)**. By analyzing key features like funding history, location, and industry sector, the model provides data-driven insights to help investors and entrepreneurs make informed decisions.

## 📊 Dataset
The dataset is sourced from **Crunchbase.com**. It contains detailed information on thousands of startups, including funding history, company milestones, and industry demographics.

## ⚙️ Data Pipeline
The `startup-prediction-eda-model.ipynb` notebook manages the end-to-end data lifecycle:

* **Renaming & Mapping:** The target column `status` is mapped to binary values where `acquired` → `1` and `closed` → `0`.
* **Correlation Analysis:** A heatmap is generated to identify features with the strongest predictive power, ensuring only relevant data influences the model.
* **Model Persistence:** Once training is complete, the optimized Random Forest model is serialized and saved as `random_forest_model.pkl` for use in the production environment.



## 🤖 Machine Learning Models
We evaluated several state-of-the-art classifiers, including XGBoost, AdaBoost, and Gradient Boosting. The **Random Forest Classifier (RFC)** was selected as the final model due to its superior performance, achieving an **accuracy score of 0.85**. 



[Image of a random forest algorithm diagram]


## 🚀 Web Deployment (Flask)
The project includes a robust Flask application (`app.py`) that serves as the bridge between the ML model and the end-user. It allows for real-time predictions through a clean web interface.

**To launch the application locally:**
1.  Ensure all dependencies are installed (`pip install flask pandas numpy scikit-learn`).
2.  Run the server:
    ```bash
    python app.py
    ```
3.  Navigate to `http://127.0.0.1:5000` in your web browser to access the prediction interface.



## ⚠️ Limitations & Future Research
* **Dataset Bias:** The model is trained on historical data; future iterations will incorporate recent post-pandemic market data to improve current relevancy.
* **Feature Expansion:** Plans are in place to integrate social media sentiment analysis and founder experience metrics to capture qualitative success factors.
* **Deep Learning:** Future research will explore Artificial Neural Networks (ANNs) to identify more complex, non-linear relationships within the startup ecosystem.

## 👨‍💻 Author
**Chaitanya Potnuri** *February 2026*
