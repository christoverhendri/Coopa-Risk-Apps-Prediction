# COPPA Risk Prediction for Mobile Applications Find IT Competition

This project focuses on predicting the COPPA (Children's Online Privacy Protection Act) compliance risk of mobile applications using advanced machine learning techniques and rigorous data preprocessing.

## 🔍 Key EDA Findings & Data Facts
Based on the raw data analysis, several critical issues were addressed:
- **Rating Anomalies**: Identified **616 records** where `userRatingCount` was 0 but `averageUserRating` was non-zero. These were force-corrected to 0 for consistency.
- **Genre Consolidation**: Merged redundant categories (e.g., merging "Book" and "Books & Reference") to reduce feature noise.
- **Outlier Management**: Capped extreme download counts exceeding 100M down to 50M to prevent model bias.
- **Geographical Masking**: Handled masked developer locations by re-categorizing them into "UNKNOWN", "MASKED", or "OTHER" based on download volume.

## 🛠️ Data Preprocessing Pipeline
1. **Feature Engineering**: 
   - Extracted `downloads_min` and `downloads_max` from string ranges.
   - Derived `appAge` by calculating absolute values to handle negative entry errors.
2. **Rule-Based Imputation**:
   - `hasPrivacyLink`: Imputed based on region-specific logic.
   - `hasTermsOfServiceLink`: Predicted using developer professional indicators (Corporate email, high-regulation countries like US/GB/DE).
3. **Advanced Imputation**:
   - Utilized an iterative **LightGBM Imputer** to fill missing values in numerical and ordinal columns, ensuring more accurate data reconstruction compared to simple mean/median imputation.

## 🧠 Model Architecture & Results
- **Algorithm**: LightGBM (Gradient Boosting Framework).
- **Optimization**: Used row-wise multi-threading for fast training and low memory overhead.
- **Classification Threshold**: 0.5 for binary classification.
- **Key Output**: 
  - `coppaRisk_predicted`: Binary classification (0 = Low Risk, 1 = High Risk).
  - `coppaRisk_proba`: Continuous probability score for fine-grained risk assessment.

## 🚀 How to Run
1. Ensure `train.csv` and `test_data_1.csv` are in the root directory.
2. Install dependencies: `pip install pandas numpy lightgbm scikit-learn tqdm`.
3. Execute the notebook to generate `submission.csv`.

---
*Note: This project was developed as part of a data science competition, focusing on domain-knowledge-driven feature engineering.*
