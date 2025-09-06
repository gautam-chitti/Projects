# **Spam Email Classifier**

## **Project Overview**

This project builds a machine learning model to effectively classify emails as "Spam" or "Not Spam" (Ham). The core of this project is based on the **Spambase dataset** from the UCI Machine Learning Repository, which contains pre-extracted features from a collection of emails. The workflow includes comprehensive exploratory data analysis (EDA), data preprocessing, model training, and evaluation, culminating in a saved, high-performing classifier that can be used for real-time predictions.

## **Table of Contents**

1. Project Structure
2. Exploratory Data Analysis (EDA) Insights
3. Data Preprocessing
4. Model Training & Results
5. How to Use
6. Future Work

## **Project Structure**

spam-classifier/  
│  
├── data/  
│ ├── raw/ \# Raw dataset  
│ ├── processed/ \# Processed data files  
│ │ ├── scaled_features.csv  
│ │ ├── pca_features.csv  
│ │ └── targets.csv  
│ └── test/
│ ├── spam_or_not_spam.csv\# External test dataset  
│  
├── models/  
│ └── best_spam_classifier.joblib \# Saved final model  
│  
├── notebooks/  
│ ├── 01_data_exploration.ipynb  
│ ├── 02_model_training.ipynb  
│ └── 03_model_evaluation.ipynb  
│  
├── reports/
│ ├── presentation.pptx
│
│
├─predict_spam.py
│  
│  
└── README.md
│── requirements.txt

## **Workflow**

The project followed a structured machine learning pipeline:

1. **Data Exploration (01_data_exploration.ipynb):** The Spambase dataset was loaded and analyzed to understand the distribution and characteristics of the features. Key relationships between word frequencies, character frequencies, capital letter usage, and the target class (Spam/Ham) were visualized.
2. **Preprocessing (01_data_exploration.ipynb):** Based on the EDA, several processed datasets were created and saved for modeling:
   - **Raw Data:** The original data with cleaned column names.
   - **Scaled Data:** Features standardized using StandardScaler to have a mean of 0 and a standard deviation of 1\.
   - **PCA Data:** Dimensionality reduction applied using Principal Component Analysis (PCA) to create a smaller feature set.
3. **Model Training (02_model_training.ipynb):** Two classification algorithms, **Logistic Regression** and **XGBoost Classifier**, were trained on all three data versions (raw, scaled, and PCA) to identify the best-performing combination.
4. **Model Selection (02_model_training.ipynb):** The models were evaluated based on accuracy and precision. The **XGBoost Classifier trained on the raw data** was selected as the best model due to its superior performance. This model was saved to the models/ directory.
5. **Inference (predict_spam.py):** An interactive command-line script was developed to load the saved model and classify new, unseen email text in real-time.
6. **Final Evaluation (03_model_evaluation.ipynb):** The performance of the saved model was validated against a completely new dataset (spam_or_not_spam.csv) to confirm its generalization capabilities.

## **Exploratory Data Analysis (EDA) Insights**

The initial analysis revealed several key indicators of spam emails:

- **Word Frequency:** Spam emails have a significantly higher frequency of words like "your," "000," "remove," "free," and "business."
- **Character Frequency:** The characters $ and \! appear much more frequently in spam emails.
- **Capitalization:** Spam emails tend to use capital letters more excessively, resulting in higher average and longest capital run lengths.

## **Data Preprocessing**

To prepare the data for modeling, the following steps were taken:

- **Feature Scaling:** StandardScaler was used to normalize the feature distributions, which is crucial for models sensitive to the scale of input data.
- **Dimensionality Reduction:** PCA was applied to reduce the 57 features down to 2 principal components for visualization and to test if a simpler model could be effective.
- **Column Name Cleaning:** Special characters (\[, \], \<) in the original dataset's column names were replaced to ensure compatibility with all libraries, particularly XGBoost.

## **Model Training & Results**

The models were systematically trained and compared. The XGBoost model trained on the raw data demonstrated the best performance.

| Data Version | Model               | Accuracy   | Precision  |
| :----------- | :------------------ | :--------- | :--------- |
| **Raw**      | **XGBoost**         | **0.9490** | **0.9365** |
| Scaled       | XGBoost             | 0.9490     | 0.9365     |
| Scaled       | Logistic Regression | 0.9294     | 0.9209     |
| Raw          | Logistic Regression | 0.9283     | 0.9160     |
| PCA          | XGBoost             | 0.8675     | 0.8414     |
| PCA          | Logistic Regression | 0.8599     | 0.8634     |

The final model, **XGBoost (Raw)**, was saved for its high accuracy and precision.

## **How to Use**

### **1\. Setup**

First, clone the repository and install the required dependencies:

``` Bash
git clone --no-checkout https://github.com/gautam-chitti/Projects.git
cd Projects
git sparse-checkout init --cone
git sparse-checkout set CapstoneProjectGautamS
git checkout

cd CapstoneProjectGautamS 
pip install \-r requirements.txt
```
### **2\. Run the Notebooks (Optional)**

To reproduce the analysis and model training, run the Jupyter notebooks in order:

1. notebooks/01_data_exploration.ipynb
2. notebooks/02_model_training.ipynb
3. notebooks/03_model_evaluation.ipynb (ensure spam_or_not_spam.csv is in data/)

### **3\. Interactive Prediction**

To classify a new email, run the interactive script from the command line:

/predict_spam.py

Paste your email text into the terminal, type ENDEMAIL on a new line, and press Enter to see the prediction.
