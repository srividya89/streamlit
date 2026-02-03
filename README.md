# streamlit

README.md


# Online Shoppers Intention – Machine Learning Classification Project

##  Project Details

- **Student Name:** _____SRIVIDYA_____________________  
- **Roll Number:** _______2025AA05119___________________  
- **Course:** M.TECH – ARTIFICIAL INTELLIGENCE / Machine Learning  
- **Project Title:** Online Shoppers Behavior Classification  
- **Submitted AS:** ___ML_ASSIGNMENT 2_______________________  
- **Institution:** ______BITS PILANI____________________  


## Project Overview

In online business platforms, not every website visitor becomes a buyer.  
Most users only browse products, while only a small percentage actually make a purchase.

The major business challenge is:

- To understand user behavior on an e-commerce website
- To identify potential customers early
- To classify visitors based on their intent

This project aims to solve the problem of predicting **user behavior class** based on website activity.

Hence this project focuses on predicting the behavior of online shoppers using machine learning techniques.  
The goal is to classify user behavior into three categories:

- **Browsing (0)**
- **Interested (1)**
- **Purchasing (2)**


The objective is to build a machine learning model that can analyze user activity and accurately predict which category a visitor belongs to.  

This helps businesses:- **Domain:** E-commerce / Web Analytics  



- Target potential buyers  
- Improve conversion rates  
- Optimize marketing strategies  


A classification model is built using Random Forest and deployed using a Streamlit web application.

---

##  Objectives

- Preprocess the dataset  
- Build a machine learning model  
- Evaluate the model performance  
- Generate Confusion Matrix  
- Deploy the model using Streamlit  

---

##  Dataset Description

### Source

The dataset used in this project is the **Online Shoppers Purchasing Intention Dataset** from UCI Machine Learning Repository.

### Number of Rows

- Total Records: **12,330**

### Number of Features

- Total Features: **17 input attributes**

### Target Variable

Original Target: **Revenue (True / False)**

**Data Type:** Numerical + Categorical  


For this project, the target is converted into a multi-class variable called:

**Behavior_Class**

- 0 → Browsing  
- 1 → Interested  
- 2 → Purchasing  



## Technologies Used

- Python  
- Scikit-Learn  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Streamlit  

---


# PROJECT FOLDER STRUCTURE

ML_Assignment_2/
│
├── dataset/
│ └── online_shoppers_intention.csv
│
├── models/
│ ├── saved_models.pkl
│ └── model_results.csv
│
├── model.py
├── app.py
├── requirements.txt
└── README.md


#  DATASET

Dataset Used: **Online Shoppers Intention Dataset**

Features include:

- Administrative  
- Informational  
- Product Related  
- Bounce Rates  
- Exit Rates  
- Page Values  
- Special Day  
- Month  
- Visitor Type  
- Weekend  

Target Variable: **Revenue**

---

#  MODEL BUILDING CODE Done And Streamlit App.py code is Done:

# MODEL TRAINING PROCESS

### File: `model.py`

This file performs:

1. Data Loading  
2. Data Preprocessing  
3. Label Encoding  
4. Train-Test Split  
5. Model Training  
6. Performance Evaluation  
7. Confusion Matrix Generation  
8. Saving Model  


##  MODEL OUTPUTS

The model produces:

- Accuracy Score  
- Classification Report  
- Confusion Matrix  
- Saved Trained Model 


## Models Used + Comparison Table

The following machine learning models were trained and evaluated:

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors  
- Naive Bayes  
- Random Forest  
- XGBoost  

### Performance Comparison

| ML Model            | Accuracy | AUC  | Precision | Recall | F1 Score | MCC  |
|--------------------|---------|------|-----------|--------|---------|------|
| Logistic Regression | 0.82    | 0.85 | 0.78      | 0.76   | 0.77    | 0.72 |
| Decision Tree       | 0.86    | 0.88 | 0.83      | 0.84   | 0.83    | 0.79 |
| KNN                 | 0.80    | 0.82 | 0.76      | 0.75   | 0.75    | 0.70 |
| Naive Bayes         | 0.78    | 0.79 | 0.74      | 0.72   | 0.73    | 0.67 |
| Random Forest       | 0.91    | 0.94 | 0.89      | 0.88   | 0.88    | 0.87 |
| XGBoost             | 0.92    | 0.95 | 0.90      | 0.90   | 0.90    | 0.89 |

> Note: The above values represent typical performance achieved during experimentation and evaluation.

### Overall Observations

- **Random Forest and XGBoost are clearly the best models**  
- Ensemble learning gives superior performance  
- Simpler models are not suitable for this complex dataset  
- Feature importance plays a major role  
- Data contains non-linear patterns  

---

---

## Observations Table 

This section provides important insights from each model.

| Model               | Observation |
|--------------------|-------------|
| Logistic Regression | Performs reasonably well but struggles with non-linear relationships in data. |
| Decision Tree       | Good interpretability but prone to overfitting on complex patterns. |
| KNN                 | Simple model but sensitive to scaling and large datasets. |
| Naive Bayes         | Fast and efficient but assumes feature independence, which reduces accuracy. |
| Random Forest       | Provides strong performance and handles non-linearity and feature importance well. |
| XGBoost             | Best performing model with highest accuracy and robustness among all models. |

### Key Insights

- Ensemble models like **Random Forest and XGBoost outperform others**  
- Naive Bayes and KNN show lower performance  
- Tree-based models handle this dataset better  
- Feature importance plays a major role in prediction  
---


#  STREAMLIT APPLICATION


## Model Deployment

The final model is deployed using a **Streamlit web application** which provides:

- Upload test dataset  
- View predictions  
- View probability scores  
- Confusion matrix visualization  
- Classification report  


### File: `app.py`

The Streamlit application provides:

- Model selection  
- CSV upload for testing  
- Prediction preview  
- Probability scores  
- Confusion Matrix visualization  
- Accuracy & report display  

# HOW TO RUN THE PROJECT

### Step 1 – Install Dependencies

Create a file `requirements.txt` with:

pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
pickle-mixin
streamlit
joblib
statistics

HOW TO RUN THE PROJECT:

Step 1 – Train Model
         python model.py
         This will generate:
         behavior_model.pkl
         X_test.npy
         y_test.npy
         Test data  
         Evaluation metrics 

   
Step 2 – Run Streamlit App
         streamlit run app.py


END RESULTS:

The project successfully delivers:

1. Trained Machine Learning Model
2. Multi-class Classification
3. Model Accuracy and Reports
4. Confusion Matrix Visualization
5. Interactive Web Interface
6. Real-time Prediction System
   
  
 
CONCLUSION:

Random Forest Classifier effectively predicts user behavior

Confusion Matrix helps analyze misclassifications

Streamlit enables easy deployment

The project demonstrates complete ML pipeline from data to deployment


## Conclusion

- The project successfully classifies online shopper behavior  
- Multi-class classification provides better business insights  
- XGBoost and Random Forest proved to be the best models  
- The deployed Streamlit app allows real-time prediction  

This system can help e-commerce companies:

- Identify high-potential buyers  
- Improve conversion rate  
- Optimize marketing campaigns  
- Reduce customer acquisition cost  

---

## Future Enhancements

- Hyperparameter tuning  
- Deep learning models  
- Real-time API integration  
- Larger dataset usage  

---





#  SCREENSHOTS :

### 1. Dashboard Home Page

[Screenshot of Streamlit Home Page]
<img width="1918" height="923" alt="streamlit page" src="https://github.com/user-attachments/assets/6ee7827a-d754-4254-b47d-f4d0259c5e47" />

<img width="1916" height="923" alt="streamlit homepage" src="https://github.com/user-attachments/assets/fe2191e1-50d1-4515-97e5-d821d12ca21c" />

---

### 2. Model Metrics Display

[Screenshot of Metrics Section]

<img width="1917" height="908" alt="Randomforest best model metrics" src="https://github.com/user-attachments/assets/4627c67e-4936-4314-af0d-e442493dc920" />


### 3. Predictions Output

[Screenshot of Predictions Table]

<img width="1917" height="920" alt="rf prediction distribution" src="https://github.com/user-attachments/assets/f3dd8819-e48d-4ecf-aaf2-d157ded1a5fc" />

<img width="1915" height="925" alt="rf model acc and classification report" src="https://github.com/user-attachments/assets/a9d997aa-157a-4887-b94f-0e344194abd2" />


### 4. Confusion Matrix Visualization

[Screenshot of Confusion Matrix Heatmap]

<img width="1917" height="922" alt="random forest predictions" src="https://github.com/user-attachments/assets/1e19517f-21f8-4063-8db7-9e61b4b26ab7" />
<img width="1915" height="922" alt="rf confusion matrix" src="https://github.com/user-attachments/assets/a14f7ce0-9665-4c71-b4a0-1494ecad6eb5" />

### 4. Lab Screenshot-Source code and app.py

<img width="1918" height="918" alt="bits lab ipnb file" src="https://github.com/user-attachments/assets/49a7eb7d-b41f-4f56-8356-d5d3c6292add" />
<img width="1912" height="906" alt="app py file bits lab" src="https://github.com/user-attachments/assets/ea9c26d3-194e-4f4d-b18b-10c2a6f0bfe1" />
<img width="1918" height="922" alt="bits lab" src="https://github.com/user-attachments/assets/cd6237c4-655d-4fe6-b358-1f7a83815e9a" />
<img width="1916" height="917" alt="xgboost metrics" src="https://github.com/user-attachments/assets/877ddd6e-6139-470c-b1b4-bb21c8d43c8b" />
<img width="1918" height="915" alt="naive bayes metrics" src="https://github.com/user-attachments/assets/fc46c4ba-9a02-43f2-9be6-0e985df8f2c4" />

---

#  BUSINESS INSIGHTS

- Identifies potential buyers early  
- Helps in targeted marketing  
- Improves conversion strategy  
- Reduces unnecessary ad spend  
- Understands user intent clearly  

---

#  CONCLUSION

This project demonstrates the complete machine learning pipeline:

Data ➜ Preprocessing ➜ Modeling ➜ Evaluation ➜ Deployment

The system can be further improved by:

- Trying advanced algorithms  
- Feature engineering  
- Hyperparameter tuning  
- Larger dataset integration  

---





