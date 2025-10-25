# 💰 Financial Risk Prediction using Logistic Regression

This project predicts the likelihood of bankruptcy or financial distress based on company risk factors, enabling better financial decision-making and early risk detection.

## 🎯 Objective
To analyze financial risk indicators and build a machine learning model that accurately predicts bankruptcy risk and highlights the most influential business factors.

## 📊 Dataset
- 250 real-world company records  
- Features include **industrial risk**, **management risk**, **financial flexibility**, **credibility**, **competitiveness**, and **operating risk**  
- Preprocessed data by handling missing values, scaling features, and encoding categorical variables

## ⚙️ Approach
- Explored relationships between various risk factors and bankruptcy likelihood  
- Trained and optimized **Logistic Regression** using **RandomizedSearchCV**  
- Performed **5-fold cross-validation** for reliable evaluation  
- Achieved a **4% improvement in test accuracy** after hyperparameter optimization

## 📈 Insights
- Identified key predictors influencing financial stability and bankruptcy risk  
- Logistic Regression effectively captured patterns in the dataset for robust predictions  
- Results support **data-driven financial risk assessment** and credit analysis

## 🧠 Tech Stack
Python • Pandas • NumPy • Scikit-learn • Matplotlib • Seaborn

## 🚀 How to Run
```bash
pip install -r requirements.txt
python Bankrupty_Prediction.py
