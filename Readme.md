# **Exploring Obesity as a Cardiovascular Disease Risk Factor – A Machine Learning Approach**

**Module:** Data Analytics and Predictive Modelling
> **Module Achievement:** Highest class score – **85%** (High Distinction)  
> **Dataset:** *Estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru, and Mexico*  
> Authors: Fabio Mendoza Palechor & Alexis de la Hoz Manotas [[Link]](https://www.sciencedirect.com/science/article/pii/S2352340919306985#bib1)  

---

## **Project Overview**  
This project applies **machine learning and statistical methods** to explore how demographic, behavioral, and dietary factors contribute to obesity and its potential progression to cardiovascular diseases.  

Using a blend of the **KDD** and **CRISP-DM** frameworks, I performed end-to-end data analysis, from preprocessing to predictive modeling by focusing on **eight key features**:

- Age  
- Weight  
- Height  
- Family history of being overweight  
- Smoking habits  
- Meal frequency  
- Alcohol consumption  
- Physical activity levels  

---

## **Tools Used**
- **Data Handling:** Excel, Python, Minitab, SPSS  
- **Machine Learning & Visualization:** Python (scikit-learn, matplotlib), Orange  
- **Statistical Analysis:** Minitab, SPSS  

---

## **Methodology**
![Project Framework](Framework.png)

### **1. Data Pre-Processing**
- Numerical Feature Analysis  
- Correlation Analysis  
- Categorical Feature Analysis  
- Proximity Analysis  

### **2. Data Processing**
- **Normality Testing**  
- **Data Discretization**  

### **3. Clustering**
- **K-Means Clustering** – Identified 5 clusters aligned with BMI Index  
  - *Silhouette Score:* 0.27  
  - *Davies-Bouldin Index:* 1.30  
- **DBScan Clustering** – Tested density-based grouping  
- **Evaluation Metrics:** Silhouette Index, Davies-Bouldin, Dunn Index  

### **4. Classification**
- **Algorithms Used:** Decision Tree (DT), K-Nearest Neighbors (KNN), Naïve Bayes, Support Vector Machine (SVM)  
- **Top Results:**  
  - DT AUC: **97%**, F1-score: **96%+**  
  - KNN AUC: **99%**, F1-score: **96%+**  
- **Metrics:** AUC, Precision, Recall, ROC Curve, Performance Curve  

### **5. Predictive Modelling**
- **Logistic Regression** – Adjusted R² = 68% (moderate explanatory power)  
- **Multivariate Linear Regression** – Investigated continuous outcome relationships  

---

## **Key Findings**
- **K-Means** effectively grouped obesity risk categories, matching BMI classification.  
- **DT and KNN** models showed strong predictive performance (AUC > 97%).  
- Age range limitation (early–late 20s) may introduce bias in generalization.  

---

## **Impact & Insights**
This study demonstrates how **data analytics** can play a pivotal role in identifying obesity risk factors and informing **preventive healthcare strategies**.  

The methodology can be adapted for broader datasets to improve **public health awareness** and **policy-making** for obesity prevention.  

---

## **Repository Structure**
```plaintext
├── data/                # Dataset files
├── notebooks/           # Jupyter Notebooks for analysis
├── scripts/             # Python scripts for ML models
├── results/             # Output graphs, evaluation metrics
└── README.md            # Project documentation
