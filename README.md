# Data-Science-Housing-price-prediction-project
# 🏠 California Housing Price Prediction

A complete end-to-end machine learning project for predicting California housing prices using the classic California Housing dataset.

This project focuses not only on achieving a good prediction score, but on understanding **why each preprocessing, modeling, and evaluation decision was made**, identifying the model's limitations, and investigating where the predictions fail.

> **Final Model:** XGBoost Regressor  
> **Test MAE:** ~$29,797  
> **Test RMSE:** ~$45,799

---

## 📌 Project Overview

The objective of this project was to build a regression model capable of predicting median house values from demographic, geographic, and housing-related features.

Rather than immediately training a powerful model, I followed a complete machine learning workflow:

**Data Inspection → Data Cleaning → Missing Value Analysis → Imputation Experiment → Outlier Investigation → Skewness Analysis → Train/Test Split → Encoding → Baseline Model → Model Comparison → Cross-Validation → Hyperparameter Tuning → Final Evaluation → Error Analysis → Feature Importance**

The goal was to treat the project as an actual ML problem rather than simply trying to maximize a metric.

---

# 🎯 Problem Statement

Given information about a California housing block, predict its **median house value**.

The available features include:

- Longitude
- Latitude
- Housing median age
- Total rooms
- Total bedrooms
- Population
- Households
- Median income
- Ocean proximity

Target:

```text
median_house_value
```

This is a **supervised regression problem** because the target is a continuous numerical value.

---

# 📊 Dataset

The project uses the classic **California Housing dataset**, based on California census data from the 1990 period.

The dataset contains:

- **20,640 observations**
- **10 columns**
- 9 predictive variables
- 1 target variable

### Dataset Structure

| Feature | Description |
|---|---|
| longitude | Geographic longitude |
| latitude | Geographic latitude |
| housing_median_age | Median age of houses in the block |
| total_rooms | Total number of rooms |
| total_bedrooms | Total number of bedrooms |
| population | Population of the block |
| households | Number of households |
| median_income | Median income |
| ocean_proximity | Distance/category relative to the ocean |
| median_house_value | Target variable |

---

# 🔎 1. Initial Data Investigation

The first step was understanding what was actually inside the dataset instead of immediately fitting a model.

The dataset contained:

- 20,640 rows
- 10 columns
- 207 missing values in `total_bedrooms`
- No missing values in the other columns

The missing values represented approximately **1% of the dataset**.

This led to the first major question:

> **How should the missing bedroom values be filled?**

I did not want to simply replace them with the mean because the number of bedrooms is related to other housing characteristics.

---

# 🧠 2. Missing Value Imputation — KNN vs Iterative

Instead of blindly choosing an imputation technique, I created an experiment.

I temporarily hid **207 known `total_bedrooms` values**, remembered their real values, and then tried to predict those hidden values.

This allowed me to compare the imputation methods against values that were actually known.

## Methods Tested

### KNN Imputation

KNN estimates a missing value by looking at similar observations.

I standardized the numerical features first and used:

```text
n_neighbors = 5
weights = "distance"
```

### Iterative Imputation

Iterative Imputation predicts missing values using relationships between the other variables.

I used a Bayesian Ridge estimator.

---

## Initial Results

| Method | MAE | RMSE |
|---|---:|---:|
| KNN | 41.76 | 71.43 |
| Iterative | 41.71 | 92.52 |

Interestingly, Iterative Imputation had a slightly lower MAE but a much worse RMSE and produced two negative predictions.

I then experimented with combining the two approaches using different weights and evaluated the stability across multiple random splits.

Although some hybrid combinations improved MAE on certain splits, the results were not consistently better than KNN.

### Final Decision

**KNN Imputation was selected.**

The important lesson was:

> A method should not be selected simply because it wins on one experiment. Stability and behavior across different samples matter.

After applying KNN to the complete dataset:

```text
Missing values before: 207
Missing values after:  0
Negative imputations:  0
```

---

# 🚨 3. Outlier Investigation

The next question was:

> **Should I remove the outliers?**

Initially, several numerical variables appeared highly skewed.

I used the IQR method to identify statistical outliers.

For example:

- `total_rooms`: 1,287 IQR outliers
- `total_bedrooms`: 1,286
- `population`: 1,196
- `households`: 1,220
- `median_income`: 681
- `median_house_value`: 1,071

At first glance, these numbers looked concerning.

But simply deleting them would have been a mistake.

Many of these observations represent legitimate large housing blocks or high-income areas.

### Important Principle Learned

> **An outlier is not automatically an error.**

I therefore followed:

**Detect → Investigate → Decide → Treat**

rather than:

**Detect → Delete**

---

# 🔍 4. Investigating Suspicious Ratios

I created diagnostic features such as:

```text
rooms_per_household
bedrooms_per_room
population_per_household
```

This revealed some extreme observations.

For example, a few rows had extremely high population-per-household ratios.

Instead of immediately deleting them, I investigated whether these represented obvious data errors.

The conclusion was to **retain the original observations** rather than make unsupported assumptions about the data.

This was an important lesson in data science:

> The model should not be "cleaned" simply to make the numbers look nicer.

---

# 📈 5. Skewness Analysis

Several variables were strongly right-skewed:

| Feature | Skewness |
|---|---:|
| population | 4.94 |
| total_rooms | 4.15 |
| total_bedrooms | 3.45 |
| households | 3.41 |
| median_income | 1.65 |
| median_house_value | 0.98 |
| housing_median_age | 0.06 |

I tested `log1p()` transformations on the heavily skewed features.

The transformation significantly reduced skewness.

However, an important lesson followed:

> **A more normally distributed feature does not automatically mean a better machine learning model.**

---

# 🌲 6. Baseline Random Forest

Before trying advanced tuning, I established a baseline.

The first Random Forest model achieved:

```text
MAE  ≈ $31,609
RMSE ≈ $48,965
```

This gave me a reference point.

Without a baseline, it would be difficult to know whether later experiments were actually improving the model.

---

# 🧪 7. Testing Log Transformation

I trained another Random Forest using log-transformed numerical features.

Results:

```text
Baseline MAE ≈ $31,609
Log-transform MAE ≈ $31,606
```

The improvement was extremely small.

Therefore, I did not keep the transformation simply because it improved skewness.

This was another important lesson:

> **Statistical improvement and model-performance improvement are not always the same thing.**

---

# 🧩 8. Feature Engineering Experiment

I created additional features:

```text
rooms_per_household
bedrooms_per_room
population_per_household
```

The idea was that ratios might provide the model with more meaningful information.

However, the feature-engineered Random Forest performed worse:

```text
Baseline MAE ≈ $31,609
Feature-engineered MAE ≈ $31,908
```

So the engineered features were not retained.

This was an important part of the project because **not every reasonable idea improves a model**.

I kept the experiment and documented the failure rather than hiding it.

---

# 🌳 9. Random Forest Tuning

I experimented with:

- Number of trees
- Maximum depth
- Minimum samples per leaf
- Maximum features

The best test-set result during the early experimentation was approximately:

```text
MAE ≈ $31,246
RMSE ≈ $48,278
```

However, repeatedly tuning against the test set creates a risk:

> The model-selection process can gradually start learning the test set.

Therefore, I stopped using the test set for model selection.

---

# 🔄 10. Cross-Validation

I switched to **5-fold cross-validation on the training data**.

This allowed model choices to be made without repeatedly looking at the final test set.

Random Forest achieved approximately:

```text
Mean CV MAE ≈ $31,955
```

I also tested Gradient Boosting:

```text
Mean CV MAE ≈ $38,356
```

Gradient Boosting performed substantially worse in this configuration.

---

# 🚀 11. XGBoost

I then introduced XGBoost as another tree-based boosting approach.

### XGBoost 100 Trees

```text
Mean CV MAE ≈ $31,904
```

This was slightly better than the Random Forest.

### XGBoost 300 Trees

```text
Mean CV MAE ≈ $31,633
```

Increasing the number of trees improved performance.

---

# ⚙️ 12. Hyperparameter Tuning

I then tested the learning rate and number of estimators.

### XGBoost

```text
n_estimators = 500
learning_rate = 0.05
```

Result:

```text
Mean CV MAE ≈ $30,772
```

This was a significant improvement.

I then tested a slower learning rate with more trees:

```text
n_estimators = 800
learning_rate = 0.03
```

Result:

```text
Mean CV MAE ≈ $30,847
```

This was worse.

So I retained:

```text
n_estimators = 500
learning_rate = 0.05
```

---

# 🌲 13. Tree Depth Experiment

I then tested tree complexity.

The default depth configuration produced:

```text
Mean CV MAE ≈ $30,772
```

Increasing the maximum depth to 8 produced:

```text
Mean CV MAE ≈ $30,158
```

This became the best cross-validated model.

The final configuration was therefore:

```text
XGBoost Regressor

n_estimators = 500
learning_rate = 0.05
max_depth = 8
```

---

# 🏆 14. Final Model Evaluation

After selecting the model using cross-validation, I trained the final model on the complete training set.

Only then did I evaluate it on the untouched test set.

### Final Results

| Metric | Result |
|---|---:|
| **MAE** | **$29,797** |
| **RMSE** | **$45,799** |

The cross-validation MAE was approximately:

```text
$30,158
```

while the final test MAE was:

```text
$29,797
```

The similarity between these values gave confidence that the model generalized reasonably well to unseen data.

---

# 📊 15. Actual vs Predicted Analysis

I plotted actual house values against predicted values.

The predictions formed a strong upward relationship with the actual values.

This indicates that the model successfully learned the general relationship between the available features and house prices.

However, prediction errors became more spread out for expensive properties.

This suggested that the model was less reliable for the upper end of the price distribution.

---

# 🔬 16. Residual Analysis

Residuals were calculated as:

```text
Residual = Actual Value - Predicted Value
```

The majority of residuals were concentrated around zero, which is a positive sign.

However, the spread of errors increased for higher-value properties.

This showed that:

> The model performs better for the majority of the market than it does for the most expensive observations.

---

# ⚠️ 17. The Most Important Dataset Limitation

This became one of the most interesting findings of the project.

The target variable contains a large number of observations at approximately:

```text
$500,001
```

This indicates a ceiling/cap in the target values.

As a result, the model cannot learn the true differences between properties whose real values may be substantially higher than the recorded maximum.

For example, if the dataset records two properties as:

```text
$500,001
$500,001
```

the model has no way of knowing whether their real market values were:

```text
$520,000
$800,000
$1,500,000
```

The training data simply does not contain that information.

This explains part of the model's weaker performance at the high-value end.

### Important conclusion

This is not necessarily a problem that can be solved by simply using a more complicated algorithm.

**The limitation originates in the data itself.**

---

# 🧠 18. Feature Importance

I examined feature importance using two different approaches.

The built-in XGBoost importance suggested strong importance for:

- Inland location
- Median income
- Latitude
- Longitude

However, I did not rely on this method alone.

I also used **permutation importance**, which measures how much model performance deteriorates when the information in a feature is randomly disrupted.

### Permutation Importance

| Feature | Importance |
|---|---:|
| **longitude** | **51,104** |
| **latitude** | **47,068** |
| **median_income** | **34,093** |
| ocean_proximity | 14,509 |
| total_rooms | 13,412 |
| population | 12,143 |
| total_bedrooms | 8,112 |
| housing_median_age | 5,042 |
| households | 4,944 |

The strongest conclusion was:

> **Geographical location and median income are the most influential predictive signals available to this model.**

This makes intuitive sense for California housing, where location has a major effect on property prices.

---

# 🤖 19. Using AI as a Learning Tool

This project was also an experiment in how I approach machine learning problems.

During development, I used an AI assistant as a **technical mentor and pair-programming aid**.

I did not simply ask for a complete project and submit the generated output.

Instead, the workflow was:

1. Encounter a problem.
2. Form a hypothesis.
3. Ask for technical guidance.
4. Implement the suggested experiment myself.
5. Run the code locally.
6. Inspect the actual output.
7. Challenge the result when necessary.
8. Compare alternatives.
9. Make the final decision based on the evidence.

For example, I initially wondered whether Iterative Imputation would be better than KNN.

Rather than accepting a theoretical answer, I created a controlled experiment where known values were artificially hidden and both methods were tested against the real values.

Later, I questioned whether repeatedly tuning against the test set was appropriate. That led to switching to cross-validation for model selection.

This process helped me learn concepts that I previously understood only theoretically, including:

- KNN imputation
- Iterative imputation
- Data leakage
- Cross-validation
- Bias vs variance
- Hyperparameter tuning
- Residual analysis
- Permutation importance
- Target-value censoring/capping
- Model limitations
- Why a statistically "better" transformation may not improve a model

The AI assistant was therefore used as a **learning and debugging partner**, while the experiments, outputs, comparisons, and final decisions were based on actual notebook results.

---

# 💡 20. Key Lessons Learned

### 1. Don't blindly remove outliers

An unusual observation may represent a real customer or real-world situation.

### 2. Better-looking data does not always produce a better model

Log transformations reduced skewness significantly, but barely improved Random Forest performance.

### 3. Feature engineering must be tested

A feature can sound useful theoretically and still make the model worse.

### 4. Don't tune against your test set repeatedly

The test set should represent the final exam.

Cross-validation should be used for model selection.

### 5. MAE and RMSE tell different stories

MAE represents typical prediction error.

RMSE gives greater weight to large mistakes.

### 6. A good model can still have important limitations

The final model performs well overall, but the target-value ceiling limits its ability to accurately model the highest-value properties.

### 7. Data quality can matter more than algorithm complexity

No algorithm can learn information that is absent from the training data.

---

# 🏢 21. Production Considerations

This model should **not** be treated as a production-ready property valuation system.

If this were being deployed for a real estate company, I would improve the system by collecting:

- More recent transaction data
- Actual property sale prices
- Exact property location
- Property size
- Number and quality of amenities
- Neighborhood characteristics
- Crime/safety statistics
- School quality
- Recent comparable-property sales
- Luxury-property-specific data
- Market trends over time

I would also monitor model performance separately across different price segments.

For example:

```text
Low-value properties
Mid-value properties
High-value properties
Luxury properties
```

If the model performs significantly worse for a particular segment, that segment could receive additional modeling or business rules.

---

# 🧭 22. What I Would Do Next

If this were turned into a real product rather than a portfolio project, the next version would focus less on squeezing another few hundred dollars out of the current MAE and more on **better data**.

A future version could explore:

- More recent datasets
- Geographic feature engineering
- Comparable-property modeling
- Separate luxury-market modeling
- Prediction intervals / uncertainty estimates
- Time-aware validation
- Model explainability
- Model monitoring
- Production deployment

---

# 🏁 Final Conclusion

This project started as a simple question:

> **"Can I predict California house prices?"**

It became a much broader machine learning exercise.

The final XGBoost model achieved:

```text
Test MAE  ≈ $29,797
Test RMSE ≈ $45,799
```

More importantly, the project demonstrated an end-to-end process of:

**Question → Experiment → Evidence → Decision → Validation**

rather than simply selecting a popular algorithm and reporting its score.

The biggest lesson was not that XGBoost was the best algorithm.

It was that **machine learning is as much about understanding the data, questioning assumptions, validating decisions, and understanding failure cases as it is about training the model itself.**

---

## 🛠️ Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

## 📁 Project Structure

```text
california-housing-price-prediction/
│
├── data/
│   └── housing.csv
│
├── notebooks/
│   └── california_housing_analysis.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## 👤 Author

**Vishal Patadiya**

This project was built as part of my journey toward becoming a stronger Machine Learning / Data Science practitioner, with an emphasis on understanding the reasoning behind each modeling decision rather than treating machine learning as a black box.
