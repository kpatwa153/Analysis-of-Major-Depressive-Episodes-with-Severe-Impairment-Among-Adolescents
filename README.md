# **📊 Analysis of Major Depressive Episodes with Severe Impairment Among Adolescents**  

## **📌 Project Overview**  
This project employs **machine learning techniques** to predict whether an adolescent is experiencing **Major Depressive Episodes with Severe Impairment (mdeSI)**. Using **logistic regression** and **random forest**, the study identifies key factors such as **age, gender, income, parental involvement, and school experience** that influence depression trends.  

The dataset, obtained from **NSDUH**, consists of **6,000 samples** with **12 features**. Data preprocessing includes **balancing classes, feature selection, and exploratory data analysis (EDA)**. The results offer insights to aid public health policies and mental health interventions.  

---

## **🚀 Key Features**  
✅ **Exploratory Data Analysis (EDA)** – Visualization of trends in depressive episodes  
✅ **Feature Selection** – Identifying key predictors using **chi-square tests and correlation analysis**  
✅ **Machine Learning Models** – Implementation of **Logistic Regression** & **Random Forest**  
✅ **Model Evaluation** – **Confusion matrices, Accuracy, Recall, AUC, and ROC curves**  
✅ **Public Health Impact** – Data-driven insights to guide **mental health interventions**  

---

## **📂 Dataset**  
- **Source:** NSDUH (National Survey on Drug Use and Health)  
- **Size:** **6,000 samples** | **12 columns**  
- **Target Variable:** `mdeSI` (Major Depressive Episodes with Severe Impairment)  
- **Key Features:**  
  - **Demographics:** Age, Gender, Race, Income  
  - **Family & Social Factors:** Parental Involvement, Household Composition, School Experience  
  - **Health Indicators:** Insurance Status, Sibling Presence, Depression History  

---

## **🔬 Methodology & Approach**  

### **1️⃣ Data Preprocessing**  
✔ Data cleaning & missing value handling  
✔ Balancing classes (50% MDE cases, 50% non-MDE cases)  
✔ Feature engineering & transformation  

### **2️⃣ Exploratory Data Analysis (EDA)**  
✔ Year-wise depression trends  
✔ Influence of gender, income, race, and parental involvement  
✔ Chi-square tests and correlation analysis for feature selection  

### **3️⃣ Machine Learning Models**  
#### **📌 Logistic Regression**  
✔ Used for **binary classification** (`mdeSI: Yes/No`)  
✔ Evaluates feature significance using **p-values & odds ratios**  
✔ Performance:  
   - Training Accuracy: **76.7%** | Recall: **83.6%** | AUC: **0.908**  
   - Testing Accuracy: **82.5%** | Recall: **73.9%** | AUC: **0.902**  

#### **📌 Random Forest**  
✔ **Ensemble learning technique** for robust classification  
✔ Feature importance ranking  
✔ Performance:  
   - Training Accuracy: **83.8%** | Recall: **76.7%** | AUC: **0.9055**  
   - Testing Accuracy: **82.8%** | Recall: **73.7%** | AUC: **0.9058**  

---

## **📊 Findings & Insights**  
- **Depression Trends:** Significant increase in cases between **2015-2016**, but stable from **2016-2017**  
- **Gender & MDE:** **75% of affected individuals** were **females**, highlighting gender-based mental health disparities  
- **Income & Depression:** Higher-income individuals showed **increased rates of MDE**, warranting further investigation  
- **Parental & School Influence:** Improved **parental involvement** and **positive school experiences** correlated with **lower depression rates** in 2017  
- **Feature Importance:** **Age, gender, income, school experience, and parental involvement** were the strongest predictors of MDE  

---

## **🛠 Technologies & Tools Used**  
- **Programming Language:** **R** 📊  
- **Libraries:** `caret`, `ggplot2`, `randomForest`, `corrplot`, `pROC`, `plotly`  
- **Machine Learning Models:** Logistic Regression, Random Forest  
- **Visualization Tools:** ggplot2, plotly, treemaps  

---

## **🛠 Installation & Setup**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/kpatwa153/Analysis-of-Major-Depressive-Episodes-with-Severe-Impairment-Among-Adolescents
```

### **2️⃣ Install Dependencies**  
Open **RStudio** and run:  
```r
install.packages(c("caret", "ggplot2", "randomForest", "corrplot", "pROC", "plotly", "tidyr", "MASS"))
```

### **3️⃣ Run the Analysis**  
Load and execute the **Patwa_FinalProject.R** script:  
```r
source("Patwa_FinalProject.R")
```

---

## **📈 Model Performance & Evaluation**  

### **📌 Confusion Matrices**  
- **Logistic Regression:**  
  ```
  Training Accuracy: 76.7% | Recall: 83.6%  
  Testing Accuracy: 82.5% | Recall: 73.9%  
  ```
- **Random Forest:**  
  ```
  Training Accuracy: 83.8% | Recall: 76.7%  
  Testing Accuracy: 82.8% | Recall: 73.7%  
  ```

### **📌 ROC & AUC Analysis**  
Both models performed well, with **AUC scores above 0.9**, indicating strong classification ability.  

---

## **🔮 Future Scope & Improvements**  
📌 **Hyperparameter tuning** – Optimize logistic regression and random forest parameters  
📌 **Additional features** – Explore external datasets for socioeconomic and psychological factors  
📌 **Deep learning models** – Apply **neural networks** for advanced prediction  
📌 **Real-world deployment** – Integrate into **mental health platforms** for predictive screening  

---

## **🤝 Contributing**  
Interested in improving the project? Contributions are welcome!  
Fork the repo, create a new branch, and submit a **pull request** with enhancements.  

---

## **📩 Contact & Acknowledgments**  
👨‍💻 **Project Author:** **Kenil Patwa**  
📜 **Acknowledgment:** **Dr. I-Ming Chiu** (Instructor)  
📧 **Email:** kenilpatwa1209@gmail.com 

🔗 **GitHub Repo:** [Project Repository](https://github.com/kpatwa153/Analysis-of-Major-Depressive-Episodes-with-Severe-Impairment-Among-Adolescents)

---

📢 **If you find this project insightful, don’t forget to ⭐ star the repo!**  

---
