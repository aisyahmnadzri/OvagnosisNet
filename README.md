# OvagnosisNet

**Classification of Ovarian Cancer Prognosis Using Deep Neural Network (DNN)**

## 📌 Overview

**OvagnosisNet** is a deep neural network (DNN) model developed for classifying ovarian cancer prognosis based on clinical and biochemical data. This project leverages machine learning and deep learning methodologies to improve the accuracy of ovarian cancer prognostication—an area of significant clinical challenge due to late-stage diagnoses and complex tumor biology.

This repository is part of my bioinformatics project for the course **Programming for Bioinformatics (SECB3203)** at [Universiti Teknologi Malaysia].

## 🎯 Objectives

* Develop a deep learning model to predict ovarian cancer outcomes.
* Identify significant clinical and molecular biomarkers using feature selection techniques.
* Evaluate the model using classification metrics such as accuracy, precision, recall, and F1 score.

## 🧬 Dataset

* Source: [NCBI PMC Article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9394434/#B29-jpm-12-01211)
* Samples: 349 patients (171 with ovarian cancer, 178 with benign tumors)
* Features: 49 clinical and biochemical indicators including blood test values and tumor markers
* Format: CSV (`ovariantotal.csv`)

## 🧪 Methodology

### 1. Data Preprocessing

* Cleaning & handling missing values
* Renaming cryptic column names for interpretability
* Standardization and normalization
* Binning (e.g. MCH levels)
* Encoding categorical features using indicator variables

### 2. Feature Engineering

* Recursive Feature Elimination (RFE)
* Descriptive statistics and ANOVA
* Pearson correlation analysis with CA125 and other biomarkers

### 3. Model Development

* Algorithms used:

  * Deep Neural Network (TensorFlow/Keras)
  * Logistic Regression, SVM, Random Forest (for benchmarking)
    
* Evaluation:

  * Accuracy, Precision, Recall, F1 Score
  * ROC curve and AUC
  * Grid Search & Hyperparameter Tuning

## 🧰 Tools and Libraries

* Python (Google Colab)
* TensorFlow
* scikit-learn
* Pandas, NumPy
* Matplotlib, Seaborn
* Jupyter Notebooks

## 📊 Results Summary

* The DNN model demonstrated high accuracy in classifying malignant vs benign ovarian tumors.
* Key predictors identified include CA125, MCH, and CEA.
* The model outperformed traditional statistical methods in robustness and generalization.

## 🏁 Conclusion

OvagnosisNet shows promise as a prognostic tool for ovarian cancer, using blood-based biomarkers and deep learning to assist in clinical decision-making. This project bridges clinical bioinformatics with machine learning for impactful, data-driven cancer prognosis.

## 📁 Repository Structure

```
OvagnosisNet/
├── data/
│   └── ovariantotal.csv
├── notebooks/
│   └── data_preprocessing.ipynb
│   └── model_training.ipynb
│   └── evaluation.ipynb
├── models/
│   └── ovagnosis_dnn.h5
├── README.md
└── requirements.txt
```

## 👥 Contributors

* Aisyah Binti Mohd Nadzri
* Thuvaarith Sivarajah

## 🧾 License

This project is for academic and research purposes only.
