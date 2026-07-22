# Telco Customer Churn Prediction using Logistic Regression

Welcome to the submission deliverables package for the **Customer Churn Prediction** project. This project implements an end-to-end machine learning pipeline to predict customer churn using the **Kaggle Telco Customer Churn dataset** and a standard **Logistic Regression** model.

The primary objective is to identify customers who are likely to cancel their telecom subscriptions. Retaining existing subscribers is highly cost-effective compared to acquiring new ones, making churn prediction a critical task for revenue protection and customer relationship management.

---

## 📁 Repository Structure

The project deliverables are organized as follows:

```text
├── Codebase/
│   ├── Customer_Churn_Prediction.ipynb       # Fully functional Jupyter Notebook containing EDA & modeling
│  
├── Dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw Kaggle Telco Customer Churn dataset
├── Generated charts/
│   ├── churn_distribution.png                # Target variable countplot
│   ├── numerical_histograms.png              # Histograms for numerical variables (tenure, charges)
│   ├── numerical_boxplots.png                # Boxplots for outlier detection
│   ├── correlation_heatmap.png               # Correlation heatmap of numerical features
│   ├── churn_by_contract.png                 # Churn rate by contract type
│   ├── churn_by_internet_service.png         # Churn rate by internet service type
│   ├── churn_by_payment_method.png           # Churn rate by payment method
│   ├── tenure_monthlycharges_by_churn.png    # Tenure and monthly charges distribution by churn
│   ├── confusion_matrix.png                  # Evaluation confusion matrix heatmap
│   ├── roc_curve.png                         # Evaluation ROC curve and ROC-AUC score
│   └── feature_importance.png                # Feature importances based on model coefficients
├── README.md                                 # Project repository overview (this file)

```

---


## 📊 Methodology & Implementation Phases

The project has been structured into six comprehensive phases matching academic and industry standards:

### Phase 1: Problem Definition & EDA
- **Business Problem:** Mitigation of subscriber churn to protect recurring revenue.
- **Data Inspection:** Analysis of 7,043 entries with 21 columns.
- **Class Imbalance:** Identified that **73.4%** of the target variable is `No Churn` and **26.6%** is `Churn`.
- **Visualizations:** Created distributions for key features, outlier detection boxplots, and correlation heatmaps showing strong linear correlation between `tenure` and `TotalCharges` (0.83).

### Phase 2: Data Preprocessing
- **Missing Values:** Converted blank spaces in `TotalCharges` to NaNs and handled the 11 missing rows appropriately by removal.
- **Categorical Encoding:** Encoded target as binary (0/1) and applied One-Hot Encoding (`drop_first=True`) to convert categorical features, yielding 30 features.
- **Feature Scaling:** Scaled numeric features using standardization (`StandardScaler`) to prevent bias in the gradient descent optimization.
- **Data Splitting:** Partitioned the data into an 80/20 train/test split with stratified sampling.

### Phase 3: Model Implementation
- Trained a binary **Logistic Regression** model using scikit-learn.
- Documented and justified the hyperparameters:
  - **Penalty:** `l2` (L2 regularization to prevent overfitting on encoded variables).
  - **Regularizer Strength C:** `1.0` (balanced inverse regularizer).
  - **Solver:** `lbfgs` (memory-efficient quasi-Newton method ideal for this dataset size).
  - **Random State:** `42` (ensuring reproducibility).

### Phase 4: Model Evaluation
The model achieved high discriminative capacity on the testing set:
- **Accuracy:** `80.38%`
- **Precision:** `64.76%` (predicted churners who actually churned)
- **Recall:** `57.49%` (actual churners successfully flagged)
- **F1 Score:** `60.91%` (balanced performance score)
- **ROC-AUC Score:** `83.57%`

### Phase 5: Interpretation & Insights
Model coefficients were extracted to identify key churn drivers:
- **Top positive drivers** (increase churn probability): `InternetService_Fiber optic` (strongest), `TotalCharges`, `StreamingTV_Yes`, `StreamingMovies_Yes`, and `PaymentMethod_Electronic check`.
- **Top negative drivers** (decrease churn probability / increase loyalty): `tenure` (strongest), long-term contract commits (`Contract_One year`, `Contract_Two year`), and security/support features (`OnlineSecurity_Yes`, `TechSupport_Yes`).

### Phase 6: Conclusion & Future Work
- **Business Actions:** Transition month-to-month customers to contracts; investigate Fiber Optic customer issues; incentivize automatic payments instead of electronic checks; promote support bundles.
- **Future Improvements:** Implement SMOTE for class imbalance, explore ensemble models (Random Forest, XGBoost), and construct usage/charges ratio features.

---
