# **Exploring Obesity as a Cardiovascular Disease Risk Factor – A Machine Learning Approach**

**Module:** Data Analytics and Predictive Modelling
> **Module Achievement:** Highest score in class (1st/150) – **85%** (High Distinction)  
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
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Feature%20Selection%20EDA%20and%20Descriptive%20Analyisis/Framework.png?raw=true" width="600" />
</p>


### **1. Data Pre-Processing**
 #### Numerical Feature Analysis  
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Feature%20Selection%20EDA%20and%20Descriptive%20Analyisis/Distribution%20of%20Numerical%20Features.png?raw=true" width="300" />
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Feature%20Selection%20EDA%20and%20Descriptive%20Analyisis/Box%20Plot%20of%20Numerical%20Values.png?raw=true" width="300" />
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Feature%20Selection%20EDA%20and%20Descriptive%20Analyisis/Pairplots%20of%20Numerical%20Features%20by%20Alcohol%20Consumption.png?raw=true" width="300" />
</p>

#### Correlation Analysis  
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Feature%20Selection%20EDA%20and%20Descriptive%20Analyisis/Coorelation%20Heatmap%20of%20Numerical%20Variables.png?raw=true" width="400" />
</p>

#### Categorical Feature Analysis  
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Feature%20Selection%20EDA%20and%20Descriptive%20Analyisis/Distribution%20of%20Categorical%20Variables.png?raw=true" width="400" />
</p>
 
#### Proximity Analysis  
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Feature%20Selection%20EDA%20and%20Descriptive%20Analyisis/Proximity%20Analysis%20of%20Gender%20and%20Age%20Gap.png?raw=true" width="600" />
</p>

### **2. Data Processing**
 
 **Normality Testing**  
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Feature%20Selection%20EDA%20and%20Descriptive%20Analyisis/Normality%20Test.png?raw=true" width="400" />
</p>


 
 **Data Discretization**  
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Feature%20Selection%20EDA%20and%20Descriptive%20Analyisis/Data%20Discretization%20of%20Categorical%20Features.png?raw=true" width="400" />
</p>

### **3. Clustering**
 #### **K-Means Clustering** – Identified 5 clusters aligned with BMI Index  
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Data%20Processing/Clustering/K%20Means%20Pictures/Elbow%20of%20K%20Means%20Clustering.png?raw=true" width="400" />
</p>

 **Cluster Performance**
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Data%20Processing/Clustering/K%20Means%20Pictures/Cluster%20Performance%20and%20Distance%20Of%20K%20Means.png?raw=true" width="600" />
</p>

*Silhouette Score:* 0.27  
*Davies-Bouldin Index:* 1.30  

<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Data%20Processing/Clustering/K%20Means%20Pictures/Evaluation%20Results.png?raw=true" width="600" />
</p>


 #### **DBScan Clustering** – Tested density-based grouping  
**Evaluation Metrics:** Silhouette Index, Davies-Bouldin, Dunn Index

<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Data%20Processing/Clustering/DB%20Scan/Validity%20Indices.png?raw=true" width= "600" />
</p>

**Clustering Results**
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Data%20Processing/Clustering/DB%20Scan/Clustering%20Result.png?raw=true" width= "600" />
</p>

### **4. Classification**
- **Algorithms Used:** Decision Tree (DT), K-Nearest Neighbors (KNN), Naïve Bayes, Support Vector Machine (SVM)  
- **Metrics:** AUC, Precision, Recall, ROC Curve, Performance Curve

**Confusion Matrix** 
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Data%20Processing/Classification/Classification%20Results.png?raw=true" width= "500" />
</p>

 **Top Results:**  
#### Decision Tree 
AUC: **97%**, F1-score: **96%+**

<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Data%20Processing/Classification/Decision%20Tree.png?raw=true" width= "500" />
</p>

 **Evaluation**
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Data%20Processing/Classification/ROC%20and%20Performance%20Curve%20of%20DT.png?raw=true" width= "500" />
</p>
 
 #### k-Nearest Neighbors (kNN)
 AUC: **99%**, F1-score: **96%+**

<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Data%20Processing/Classification/KNN%20Result.png?raw=true" width= "500" />
</p>
    
 *Feature Ranking*
 
 <p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Data%20Processing/Classification/Feature%20Ranking.png?raw=true" width= "500" />
</p>
 
 
### **5. Predictive Modelling**
- **Logistic Regression** – Adjusted R² = 68% (moderate explanatory power)  
- **Multivariate Linear Regression** – Investigated continuous outcome relationships  

**Logistic Regression with Backward Elimination**
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Data%20Processing/Regression%20Analysis/Logistic%20Regression%20with%20Backward%20Elimination.png?raw=true" width="500" />
</p>

**Final Regression Result**
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Data%20Processing/Regression%20Analysis/Final%20Regression%20Result.png?raw=true" width="500" />
</p>

**ROC of Final Regression**
<p align="center">
  <img src="https://github.com/Shadman-Rummat/Exploring-Obesity-A-Machine-Learning-Approach/blob/main/Data%20Processing/Regression%20Analysis/ROC%20of%20Logistic%20Regression.png?raw=true" width="500" />
</p>


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
