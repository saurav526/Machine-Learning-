import re
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, Binarizer, LabelEncoder, OrdinalEncoder,  OneHotEncoder, PolynomialFeatures, PowerTransformer, QuantileTransformer
)
from sklearn.model_selection import ( train_test_split, cross_val_score, KFold, StratifiedKFold,TimeSeriesSplit)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
)
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer
)
from sklearn.metrics import pairwise_distances

# ------------------------------------------------------------
# 0. DEMO DATA
# ------------------------------------------------------------
df = pd.DataFrame({
    "age": [21, 25, np.nan, 35, 120, 29, 29],
    "salary": [30000, 45000, 50000, np.nan, 900000, 52000, 52000],
    "city": ["Pune", "Mumbai", "Pune", "Delhi", None, "Mumbai", "Mumbai"],
    "education": ["UG", "PG", "UG", "PhD", "UG", "PG", "PG"],
    "purchased": ["No", "Yes", "No", "Yes", "Yes", "No", "No"],
    "signup_date": ["2025-01-01", "2025-02-15", "bad-date", "2025-04-10",
                    "2025-05-20", "2025-06-01", "2025-06-01"],
    "review": ["Great product!", "not good", "Excellent value", None,
               "Too expensive", "great product!", "great product!"]
})

print("Original data:")
print(df.head())


# ============================================================
# 1. DATA INSPECTION
# ============================================================

df.head()                 # Displays the first rows to preview the dataset.
df.tail()                 # Displays the last rows to inspect the end of the dataset.
df.sample(3, random_state=42)  # Randomly samples rows to inspect representative records.
df.shape                  # Returns the number of rows and columns.
df.columns                # Lists all column names.
df.dtypes                 # Shows each column's data type.
df.info()                 # Summarizes column types, non-null counts, and memory usage.
df.describe(include="all")  # Summarizes numerical and categorical columns.
df.isna().sum()           # Counts missing values in each column.
df.isna().mean() * 100    # Calculates the percentage of missing values per column.
df.nunique()              # Counts distinct values in each column.
df["city"].value_counts(dropna=False)  # Counts category frequencies, including missing values.
df.duplicated().sum()     # Counts fully duplicated rows.
df.corr(numeric_only=True)  # Calculates correlations among numerical columns.


# ============================================================
# 2. DATA TYPE CONVERSION
# ============================================================

df["age"] = pd.to_numeric(df["age"], errors="coerce")  # Converts values to numbers and turns invalid values into NaN.
df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")  # Parses dates and marks invalid dates as NaT.
df["city"] = df["city"].astype("category")  # Stores repeated labels as a categorical data type.
# df["id"] = df["id"].astype("string")  # Converts an identifier column to string without treating it as a measurement.
# df["amount"] = df["amount"].astype("float64")  # Converts a numeric column to floating-point values.


# ============================================================
# 3. DUPLICATE HANDLING
# ============================================================

df_no_duplicates = df.drop_duplicates()  # Removes rows that are identical across all columns.
df_keep_first = df.drop_duplicates(subset=["city", "salary"], keep="first")  # Keeps the first row for each selected key combination.
df_keep_last = df.drop_duplicates(subset=["city", "salary"], keep="last")  # Keeps the last row for each selected key combination.


# ============================================================
# 4. MISSING VALUE HANDLING / IMPUTATION
# ============================================================

df_missing_removed = df.dropna()  # Removes every row containing at least one missing value.
df_column_removed = df.dropna(axis=1, thresh=int(0.7 * len(df)))  # Removes columns with fewer than 70% observed values.
df["age_fill_mean"] = df["age"].fillna(df["age"].mean())  # Replaces missing numeric values with the column mean.
df["age_fill_median"] = df["age"].fillna(df["age"].median())  # Replaces missing numeric values with the column median.
df["city_fill_mode"] = df["city"].astype("object").fillna(df["city"].mode().iloc[0])  # Replaces missing categories with the most frequent category.
df["review_fill_constant"] = df["review"].fillna("Unknown")  # Replaces missing text with a fixed label.
df["age_fill_forward"] = df["age"].ffill()  # Fills missing values with the previous observed value, useful for ordered data.
df["age_fill_backward"] = df["age"].bfill()  # Fills missing values with the next observed value, useful for ordered data.
df["age_fill_interpolation"] = df["age"].interpolate(method="linear")  # Estimates missing numeric values between neighboring observations.
# Fit imputers on training data only to avoid leakage.
mean_imputer = SimpleImputer(strategy="mean")  # Learns training-column means and uses them to fill numeric missing values.
median_imputer = SimpleImputer(strategy="median")  # Learns training-column medians for robust numeric imputation.
mode_imputer = SimpleImputer(strategy="most_frequent")  # Learns the most frequent training value for each column.
constant_imputer = SimpleImputer(strategy="constant", fill_value=-1)  # Replaces missing values with a chosen sentinel value.
knn_imputer = KNNImputer(n_neighbors=3)  # Estimates missing numeric values using nearby rows with similar observed features.
# Example: X_imputed = median_imputer.fit_transform(X_train); X_test_imputed = median_imputer.transform(X_test)


# ============================================================
# 5. OUTLIER DETECTION AND TREATMENT
# ============================================================

numeric = df[["age", "salary"]].copy()  # Selects numeric columns for outlier analysis.
q1 = numeric.quantile(0.25)  # Computes first quartiles for the IQR method.
q3 = numeric.quantile(0.75)  # Computes third quartiles for the IQR method.
iqr = q3 - q1  # Measures the middle 50% spread of each numeric column.
iqr_mask = ~((numeric < (q1 - 1.5 * iqr)) | (numeric > (q3 + 1.5 * iqr))).any(axis=1)  # Flags rows without IQR-defined outliers.
numeric_iqr_filtered = numeric.loc[iqr_mask]  # Removes rows flagged as IQR outliers.
z_scores = numeric.apply(lambda col: np.abs(stats.zscore(col, nan_policy="omit")))  # Computes absolute z-scores to measure distance from the mean.
zscore_mask = (z_scores < 3).all(axis=1)  # Flags rows whose observed numeric values are within three standard deviations.
numeric_z_filtered = numeric.loc[zscore_mask]  # Keeps rows that pass the z-score rule.
numeric_winsorized = numeric.clip(lower=numeric.quantile(0.01), upper=numeric.quantile(0.99), axis=1)  # Caps each column at its 1st and 99th percentiles.
iso = IsolationForest(contamination=0.05, random_state=42)  # Creates an anomaly detector that isolates unusual observations.
iso_labels = iso.fit_predict(numeric.fillna(numeric.median()))  # Labels observations as inliers (1) or outliers (-1).
lof = LocalOutlierFactor(n_neighbors=3, contamination=0.05)  # Creates a local-density method for identifying unusual observations.
lof_labels = lof.fit_predict(numeric.fillna(numeric.median()))  # Detects rows whose local density differs strongly from their neighbors.
# Choose a method based on the data and domain; unusual values may be valid, not errors.


# ============================================================
# 6. CATEGORICAL ENCODING
# ============================================================

sample_categories = pd.DataFrame({"city": ["Pune", "Mumbai", "Delhi", "Pune"]})
one_hot = pd.get_dummies(sample_categories, columns=["city"], dtype=int)  # Creates one binary column for each category.
label_encoder = LabelEncoder()  # Encodes target labels as integer classes; generally not for nominal input features.
encoded_target = label_encoder.fit_transform(pd.Series(["No", "Yes", "No"]))  # Converts a classification target into integer class IDs.
ordinal_encoder = OrdinalEncoder(categories=[["UG", "PG", "PhD"]])  # Encodes ordered education levels according to a supplied order.
education_encoded = ordinal_encoder.fit_transform(df[["education"]])  # Transforms education categories into ordinal numeric codes.
# For production, fit OneHotEncoder(handle_unknown="ignore") on training data and reuse it on test data.
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # Encodes categories and safely handles unseen test categories.
# X_cat_train_encoded = ohe.fit_transform(X_cat_train); X_cat_test_encoded = ohe.transform(X_cat_test)


# ============================================================
# 7. FEATURE SCALING AND NORMALIZATION
# ============================================================

X_demo = pd.DataFrame({"age": [20, 30, 40], "salary": [20000, 50000, 80000]})
standard_scaler = StandardScaler()  # Centers features at mean zero and scales them to unit variance.
X_standard = standard_scaler.fit_transform(X_demo)  # Learns scaling parameters and standardizes the demo data.
minmax_scaler = MinMaxScaler(feature_range=(0, 1))  # Maps each feature into a specified range, commonly zero to one.
X_minmax = minmax_scaler.fit_transform(X_demo)  # Applies min-max scaling to the demo data.
robust_scaler = RobustScaler()  # Scales using the median and interquartile range to reduce outlier influence.
X_robust = robust_scaler.fit_transform(X_demo)  # Applies robust scaling to the demo data.
maxabs_scaler = MaxAbsScaler()  # Divides each feature by its maximum absolute value while preserving sparsity.
X_maxabs = maxabs_scaler.fit_transform(X_demo)  # Applies max-absolute scaling to the demo data.
normalizer = Normalizer(norm="l2")  # Scales each row vector to unit length rather than scaling each column.
X_normalized = normalizer.fit_transform(X_demo)  # Applies row-wise L2 normalization to the demo data.
binarizer = Binarizer(threshold=0.5)  # Converts values above a threshold to one and the rest to zero.
X_binary = binarizer.fit_transform(np.array([[0.2], [0.8]]))  # Demonstrates threshold-based binarization.
power_transformer = PowerTransformer(method="yeo-johnson")  # Applies a power transform to make numeric distributions more Gaussian-like.
X_power = power_transformer.fit_transform(X_demo)  # Applies the Yeo-Johnson transformation to demo features.
quantile_transformer = QuantileTransformer(output_distribution="normal", n_quantiles=3, random_state=42)  # Maps feature quantiles to a normal distribution.
X_quantile = quantile_transformer.fit_transform(X_demo)  # Applies quantile transformation to demo features.
# Fit all scalers on X_train only, then call transform() on X_test to prevent data leakage.


# ============================================================
# 8. DISTRIBUTION TRANSFORMS
# ============================================================

positive_values = pd.Series([1, 2, 5, 10, 100])
log_values = np.log1p(positive_values)  # Compresses right-skewed nonnegative values using log(1 + x).
sqrt_values = np.sqrt(positive_values)  # Reduces right skew with a square-root transform for nonnegative values.
reciprocal_values = 1 / positive_values  # Applies a reciprocal transform to positive values, reversing their scale.
# Box-Cox requires strictly positive data; use PowerTransformer(method="box-cox") only when that condition holds.
# Yeo-Johnson supports zero and negative values through PowerTransformer(method="yeo-johnson").



# 9. FEATURE ENGINEERING
# ============================================================

features = pd.DataFrame({"height_cm": [170, 180, 160], "weight_kg": [65, 80, 55], "price": [100, 200, 150]})
features["bmi"] = features["weight_kg"] / (features["height_cm"] / 100) ** 2  # Creates BMI from weight and height.
features["price_per_10"] = features["price"] / 10  # Creates a domain-relevant scaled price feature.
features["height_squared"] = features["height_cm"] ** 2  # Adds a nonlinear squared feature.
features["height_weight_interaction"] = features["height_cm"] * features["weight_kg"]  # Adds an interaction between height and weight.
poly = PolynomialFeatures(degree=2, include_bias=False)  # Generates polynomial and interaction terms up to degree two.
poly_features = poly.fit_transform(features[["height_cm", "weight_kg"]])  # Builds polynomial features from selected columns.
features["price_bin"] = pd.cut(features["price"], bins=[0, 120, 180, np.inf], labels=["Low", "Medium", "High"])  # Converts a continuous price into interpretable ranges.
features["price_quantile_bin"] = pd.qcut(features["price"], q=3, labels=["Low", "Medium", "High"], duplicates="drop")  # Splits values into quantile-based groups.
features["height_rank"] = features["height_cm"].rank(method="average")  # Replaces values with their relative ranks.
features["price_log"] = np.log1p(features["price"])  # Adds a log-transformed price feature.


# ============================================================
# 10. DATE AND TIME FEATURES
# ============================================================

dates = pd.to_datetime(pd.Series(["2025-01-15", "2025-06-21", "2025-12-31"]))
date_features = pd.DataFrame({"date": dates})
date_features["year"] = date_features["date"].dt.year  # Extracts the calendar year.
date_features["month"] = date_features["date"].dt.month  # Extracts the month number.
date_features["day"] = date_features["date"].dt.day  # Extracts the day of the month.
date_features["day_of_week"] = date_features["date"].dt.dayofweek  # Encodes Monday as zero through Sunday as six.
date_features["quarter"] = date_features["date"].dt.quarter  # Extracts the calendar quarter.
date_features["is_weekend"] = date_features["date"].dt.dayofweek >= 5  # Flags Saturdays and Sundays.
date_features["day_of_year"] = date_features["date"].dt.dayofyear  # Extracts the ordinal day within the year.
date_features["month_sin"] = np.sin(2 * np.pi * date_features["month"] / 12)  # Cyclically encodes month position with a sine feature.
date_features["month_cos"] = np.cos(2 * np.pi * date_features["month"] / 12)  # Cyclically encodes month position with a cosine feature.
# Time series: ts.shift(1)  # Creates a lag feature using the previous observation.
# Time series: ts.rolling(7).mean()  # Computes a trailing seven-observation rolling mean.
# Time series: ts.diff()  # Computes the change from the previous observation.
# Time series: ts.pct_change()  # Computes the percentage change from the previous observation.


# ============================================================
# 11. TEXT PREPROCESSING
# ============================================================

text_series = pd.Series(["The Product is GREAT!!!", "Not good at all.", None, "AI & Data Science"])
text_clean = text_series.fillna("").str.lower()  # Replaces missing text with empty strings and lowercases characters.
text_clean = text_clean.str.strip()  # Removes whitespace at the beginning and end of each text.
text_clean = text_clean.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)  # Removes URLs from text.
text_clean = text_clean.str.replace(r"<[^>]+>", " ", regex=True)  # Removes simple HTML tags from text.
text_clean = text_clean.str.replace(r"[^a-z0-9\s]", " ", regex=True)  # Removes punctuation and non-alphanumeric characters for English text.
text_clean = text_clean.str.replace(r"\s+", " ", regex=True).str.strip()  # Collapses repeated whitespace into single spaces.
count_vectorizer = CountVectorizer()  # Converts a collection of documents into word-count vectors.
count_matrix = count_vectorizer.fit_transform(text_clean)  # Builds a document-term count matrix.
tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))  # Weights terms by importance and includes unigrams and bigrams.
tfidf_matrix = tfidf_vectorizer.fit_transform(text_clean)  # Builds TF-IDF document vectors.
# Tokenization: nltk.word_tokenize(text)  # Splits text into tokens when the relevant NLTK tokenizer data is installed.
# Stopword removal: remove tokens in a language-specific stopword list to reduce common words.
# Stemming: PorterStemmer().stem(token)  # Reduces words to crude stems without requiring dictionary-valid words.
# Lemmatization: WordNetLemmatizer().lemmatize(token)  # Reduces words to dictionary forms when WordNet resources are installed.
# IMPORTANT: For sentiment analysis, preserve negations such as "not" because they can reverse meaning.


# ============================================================
# 12. FEATURE SELECTION
# ============================================================

variance_selector = VarianceThreshold(threshold=0.0)  # Removes features that are constant in the training data.
X_variance_selected = variance_selector.fit_transform(X_demo)  # Keeps columns that exceed the variance threshold.
# SelectKBest(f_classif, k=5)  # Selects the five features with the strongest ANOVA association with a classification target.
# SelectKBest(mutual_info_classif, k=5)  # Selects features with high estimated mutual information with the target.
# Correlation filtering: remove one of two highly correlated predictors when redundancy is undesirable.
# Domain selection: retain features justified by subject-matter knowledge and availability at prediction time.


# ============================================================
# 13. DIMENSIONALITY REDUCTION
# ============================================================

pca = PCA(n_components=2, random_state=42)  # Defines principal component analysis to project features into two dimensions.
X_pca = pca.fit_transform(StandardScaler().fit_transform(X_demo))  # Scales features and projects them into two principal components.
pca.explained_variance_ratio_  # Reports the fraction of variance explained by each retained principal component.
# Fit PCA on training data only and apply the fitted transformation to validation and test data.


# ============================================================
# 14. CLASS IMBALANCE HANDLING
# ============================================================

# In train_test_split(..., stratify=y), stratification preserves class proportions across splits.
# class_weight="balanced"  # Makes many sklearn classifiers penalize errors on minority classes more strongly.
# from imblearn.over_sampling import RandomOverSampler  # Imports random minority-class oversampling (requires imbalanced-learn).
# from imblearn.over_sampling import SMOTE  # Imports synthetic minority oversampling for numeric feature spaces.
# from imblearn.under_sampling import RandomUnderSampler  # Imports majority-class undersampling.
# Never resample before splitting; resample only training data to prevent leakage.


# ============================================================
# 15. TRAIN / VALIDATION / TEST SPLITTING
# ============================================================

X = pd.DataFrame({"feature_1": range(100), "feature_2": np.arange(100) * 2})
y = pd.Series([0, 1] * 50)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)  # Holds out 20% of rows while preserving class balance.
X_train2, X_temp, y_train2, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)  # Creates a 70% training split and a 30% temporary split.
X_val, X_test2, y_val, y_test2 = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)  # Splits the temporary portion evenly into validation and test sets.
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Defines five-fold cross-validation for ordinary independent samples.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Defines stratified folds for classification tasks.
tscv = TimeSeriesSplit(n_splits=5)  # Defines forward-chaining splits that preserve time order.
# For time series, split chronologically instead of randomly to avoid future-to-past leakage.


# ============================================================
# 16. PIPELINE: RECOMMENDED END-TO-END TABULAR PREPROCESSING
# ============================================================

# Example structure for a real dataset:
# X = df.drop(columns=["target"])
# y = df["target"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# numeric_features = X_train.select_dtypes(include=np.number).columns
# categorical_features = X_train.select_dtypes(include=["object", "category", "bool"]).columns
#
# numeric_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),  # Fills missing numeric values using training medians.
#     ("scaler", StandardScaler()),  # Standardizes numeric columns using training statistics.
# ])
#
# categorical_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),  # Fills missing categories using training modes.
#     ("encoder", OneHotEncoder(handle_unknown="ignore")),  # Encodes categories and tolerates unseen test categories.
# ])
#
# preprocessor = ColumnTransformer([
#     ("num", numeric_pipeline, numeric_features),
#     ("cat", categorical_pipeline, categorical_features),
# ])
#
# X_train_processed = preprocessor.fit_transform(X_train)  # Learns preprocessing rules only from training data.
# X_test_processed = preprocessor.transform(X_test)  # Applies the learned rules to test data without refitting.
#
# from sklearn.linear_model import LogisticRegression
# full_pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("model", LogisticRegression(max_iter=1000)),
# ])  # Combines preprocessing and prediction into one reusable, leakage-resistant workflow.
# full_pipeline.fit(X_train, y_train)  # Fits preprocessing and the estimator using training data only.
# predictions = full_pipeline.predict(X_test)  # Produces predictions for untouched test rows.


# ============================================================
# 17. IMAGE PREPROCESSING (OPTIONAL: OPENCV / PIL)
# ============================================================

# image = cv2.imread("image.jpg")  # Reads an image from disk in BGR channel order.
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converts OpenCV's BGR image to RGB.
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converts a color image to grayscale.
# image_resized = cv2.resize(image, (224, 224))  # Resizes an image to the input size expected by a model.
# image_normalized = image_resized.astype("float32") / 255.0  # Scales 8-bit pixel intensities into the [0, 1] range.
# image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)  # Reduces high-frequency noise using Gaussian smoothing.
# edges = cv2.Canny(image_gray, 100, 200)  # Detects strong image edges using Canny edge detection.
# _, binary_image = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)  # Converts grayscale pixels to black or white using a threshold.
# Image augmentation may include flips, rotations, crops, and color jitter applied to training images only.


# ============================================================
# 18. DATA VALIDATION AND CONSISTENCY
# ============================================================

df["city_clean"] = df["city"].astype("string").str.strip().str.title()  # Standardizes category whitespace and capitalization.
df["review_clean"] = df["review"].astype("string").str.strip()  # Removes leading and trailing whitespace from text.
# df = df.rename(columns=str.lower)  # Standardizes column names to lowercase.
# df = df.rename(columns=lambda c: re.sub(r"\W+", "_", c.strip().lower()))  # Makes column names lowercase and underscore-separated.
# df = df[df["age"].between(0, 120) | df["age"].isna()]  # Filters impossible ages while retaining missing values for separate handling.
# df = df[df["salary"] >= 0]  # Removes negative salaries when the domain rules say they are invalid.
# df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"})  # Maps inconsistent labels to canonical category names.
# Assert rules: assert df["id"].is_unique  # Checks that identifier values are unique.
# Assert rules: assert df["price"].ge(0).all()  # Checks that all prices are nonnegative.
# Assert rules: assert df["event_date"].notna().all()  # Checks that required dates contain no missing values.


# ============================================================
# 19. SAVE PROCESSED DATA
# ============================================================

# df.to_csv("processed_data.csv", index=False)  # Exports the processed dataframe to CSV without writing the index.
# df.to_parquet("processed_data.parquet", index=False)  # Saves data in efficient columnar Parquet format (requires pyarrow or fastparquet).
# df.to_excel("processed_data.xlsx", index=False)  # Exports the processed dataframe to an Excel workbook.
# joblib.dump(full_pipeline, "preprocessing_model.joblib")  # Saves a fitted sklearn pipeline for reuse (requires joblib).


# ============================================================
# KEY RULES
# ============================================================
# 1. Split data before learning imputation, scaling, encoding, feature selection, or resampling rules.
# 2. Fit preprocessing on training data only; transform validation and test data with the same fitted objects.
# 3. Choose methods based on the data type, distribution, model, and domain—not by applying every technique.
# 4. Do not remove all outliers automatically; first check whether they are errors or valid rare cases.
# 5. Preserve a copy of raw data and document every transformation.
# 6. For text, images, audio, and time series, use modality-specific preprocessing suited to the model.

print("Preprocessing reference loaded successfully.")

