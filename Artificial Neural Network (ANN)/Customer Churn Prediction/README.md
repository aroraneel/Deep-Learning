# Customer Churn Prediction using Artificial Neural Network (ANN)

## 📌 Overview
This project predicts whether a customer will leave a bank (**churn**) using an **Artificial Neural Network (ANN)** built with TensorFlow/Keras.

Customer churn prediction is an important business problem in banking, helping companies retain customers by identifying high-risk clients early.

The model is trained on structured customer data and evaluated using accuracy and classification metrics.

---

## 🎯 Objectives
- Perform exploratory analysis on customer churn data  
- Preprocess categorical and numerical features  
- Build a deep learning ANN classification model  
- Predict customer exit behavior (`Exited`)  
- Evaluate model performance using:
  - Accuracy Score  
  - Classification Report  

---

## 📁 Dataset Information
- Dataset: **Customer Churn Modelling Dataset**
- Target Variable:
  - `Exited` (1 = Customer left, 0 = Customer stayed)

### Input Features Include:
- Credit Score  
- Geography  
- Gender  
- Age  
- Balance  
- Number of Products  
- Estimated Salary  
- Tenure  

---

## 🔍 Methodology

### 1. Data Loading
- Dataset is loaded using Pandas  
- Features are selected excluding irrelevant columns  

---

### 2. Exploratory Data Analysis (EDA)
- Countplot is used to visualize churn distribution  
- Correlation heatmap is generated for numeric features  

---

### 3. Data Preprocessing

#### One-Hot Encoding
- Categorical variables are converted into numeric form using:
  - `pd.get_dummies()`

#### Train-Test Split
- Data is split into:
  - 80% Training  
  - 20% Testing  
- Stratified split ensures balanced churn distribution  

#### Feature Scaling
- Standardization is applied using `StandardScaler`  
- Scaling improves ANN training performance  

---

## 🧠 Model Development (ANN)

### Neural Network Architecture
- Input Layer  
- Dense Layer (128 neurons, ReLU)  
- Dense Layer (128 neurons, ReLU)  
- Output Layer (1 neuron, Sigmoid)

Activation Functions:
- ReLU → Hidden layers  
- Sigmoid → Binary classification output  

---

## ⚙️ Model Training
- Loss Function: `binary_crossentropy`  
- Optimizer: Adam (`learning_rate = 0.001`)  
- Epochs: 10  
- Batch Size: 32  
- Validation performed on test data  

---

## 📊 Model Evaluation

### Metrics Used
- Classification Report (Precision, Recall, F1-score)
- Accuracy Score

Predictions are converted into binary labels using a 0.5 threshold.

---

## 🛠️ Tools & Libraries
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- TensorFlow / Keras  

---

## 🚀 Applications
- Banking customer retention strategies  
- Churn risk prediction systems  
- Business decision support  
- Deep learning classification on structured datasets  

---

## 👤 Author
**Neel Arora**  
BCA Undergraduate | Data Science & Machine Learning Enthusiast  

---
