# Customer Churn Analysis & Prediction

## Project overview
This project analyzes customer churn in a telecommunications sample dataset and develops classification models to estimate whether a customer is likely to churn.

## Business objective
Understand patterns associated with customer churn, evaluate predictive models, and translate supported findings into practical retention recommendations.

## Dataset
- Name: IBM Telco Customer Churn sample dataset
- Target: `Churn`
- Source: https://github.com/IBM/telco-customer-churn-on-icp4d
- Alternative public mirror: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Verified structure of the supplied file: **7,043 rows × 21 columns**, no duplicates, 11 blank `TotalCharges` values (all customers with `tenure = 0`).

## Key results (all produced by the notebook)
- Churn rate: **26.5%** (1,869 of 7,043 customers).
- Highest-churn groups: month-to-month contracts (42.7%), tenure ≤ 12 months (47.4%), fiber optic (41.9%), electronic-check payers (45.3%). Two-year contracts: 2.8%.
- Six models compared (stratified 80/20 split, 5-fold CV tuning). Test ROC-AUC 0.83–0.85; the top three are statistically tied.
- **Selected model: Logistic Regression** (test ROC-AUC 0.841, recall 0.78 / precision 0.50 at threshold 0.50; recall 0.90 / precision 0.45 at the chosen 0.35 threshold). Selected for stability, transparency and speed — not for accuracy.

| Model | Test ROC-AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| Gradient Boosting | 0.846 | 0.503 | 0.660 | 0.571 | 0.799 |
| Random Forest | 0.844 | 0.791 | 0.517 | 0.625 | 0.748 |
| **Logistic Regression** | 0.841 | 0.783 | 0.504 | 0.614 | 0.738 |
| Decision Tree | 0.836 | 0.781 | 0.505 | 0.613 | 0.739 |
| SVM (RBF) | 0.834 | 0.626 | 0.594 | 0.609 | 0.787 |
| KNN | 0.831 | 0.575 | 0.592 | 0.583 | 0.782 |

(Threshold 0.50 for all rows. Full table with PR-AUC and CV AUC: `outputs/model_comparison.csv`.)

## Tools
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook

## Repository structure
```text
customer-churn-analysis/
├── data/
│   ├── raw/Telco-Customer-Churn.csv         # original file, never modified
│   ├── processed/telco_churn_cleaned.csv    # cleaned output
│   └── README.md
├── notebooks/
│   └── Customer_Churn_Analysis.ipynb        # executed, end-to-end analysis
├── src/
│   ├── churn_utils.py                       # shared helpers: cleaning, features, preprocessing, metrics
│   ├── data_preprocessing.py                # raw -> processed CSV, train/test split
│   ├── model_training.py                    # trains 6 models, writes comparison table + best model
│   └── prediction.py                        # scores customers with the saved model
├── models/
│   └── best_model.joblib                    # Logistic Regression pipeline + chosen threshold (0.35)
├── outputs/
│   ├── visualizations/                      # PNG charts
│   ├── model_comparison.csv
│   └── predictions.csv                      # scores for all 7,043 customers (see 'dataset_split')
├── dashboard/
│   ├── app.py                               # Streamlit dashboard
│   └── dashboard.html                       # standalone interactive page (no install needed)
├── reports/
│   ├── Final_Report.pdf                     # method, results, limitations
│   └── Business_Recommendations.pdf         # segments, actions, how to use the model
├── requirements.txt
├── .gitignore
└── README.md
```

## Workflow
1. Data understanding and quality checks
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Feature engineering
5. Train/test split and model training
6. Model evaluation and comparison
7. Interpretation, limitations, and business recommendations

## How to run
1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Notebook** (full analysis, about 2-3 minutes; SVM and forest searches are slowest):
   ```bash
   jupyter notebook notebooks/Customer_Churn_Analysis.ipynb
   ```
5. **Or run the pipeline as scripts** from the project root (they reproduce the notebook's numbers):
   ```bash
   python src/data_preprocessing.py     # clean data -> data/processed/
   python src/model_training.py         # compare models, save models/best_model.joblib
   python src/prediction.py             # score customers -> outputs/predictions.csv
   python src/prediction.py --input new_customers.csv --output scored.csv   # same columns as the raw file
   ```
6. **Dashboard:**
   ```bash
   streamlit run dashboard/app.py       # or just open dashboard/dashboard.html in a browser
   ```

**About `predictions.csv`:** about 80% of rows have `dataset_split = train`, meaning the model was fitted on them, so their scores look better than real life. Judge performance on the `test` rows only. Risk tiers: High >= 0.75, Medium >= 0.35, Low below. Scores rank customers; they are not calibrated probabilities.

## Evaluation
Report appropriate metrics such as precision, recall, F1-score, ROC-AUC, and confusion matrix. Do not choose a model using accuracy alone. Explain the trade-off between false positives and false negatives.

In this project: models are compared on ROC-AUC (cross-validated and test), PR-AUC, recall, precision and F1; the decision threshold for the chosen model is selected on out-of-fold training predictions (maximising F2, i.e. recall weighted above precision — an assumption to confirm with the client) and then evaluated once on the test set.

## Limitations
- This is a historical sample dataset, not a live company's customer database.
- Associations do not establish causation.
- Model performance on this dataset does not guarantee performance on new or current customers.
- The model is a decision-support demonstration, not an automated retention decision system.
- The file has no timestamps or churn reasons; `tenure` of churned customers stops at the moment they leave, so some predictive signal is partly built into how the data was recorded.
- Predicted scores are not calibrated probabilities (class weights were used); a fairness review (e.g. `SeniorCitizen`) has not been performed.

## Data and privacy
Do not commit confidential, personal, or client-owned data to a public repository. If the client provides private data, keep it in an approved secure location and add it to `.gitignore`. Never commit passwords, API keys, or access tokens.

## Author
[Your name / GitHub profile]
