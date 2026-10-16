

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, roc_curve, auc,
                            roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

#Configure visualization settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*30)
print("TELCO CUSTOMER CHURN ANALYSIS AND PREDICTION")
print("="*30)
print("\n")


#STEP 1: DESCRIPTIVE STATISTICS


print("STEP 1: DESCRIPTIVE STATISTICS")


#Loading dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("\n1.1 Dataset Overview:")
print(f"Number of data points (rows): {df.shape[0]}")
print(f"Number of features (columns): {df.shape[1]}")
print(f"Number of descriptive features: {df.shape[1] - 1}")

print("\n1.2 First few rows:")
print(df.head(10))

print("\n1.3 Dataset Information:")
print(df.info())

#Identifying feature types
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print("\n1.4 Feature Types:")
print(f"Numerical ({len(numerical_features)}): {numerical_features}")
print(f"Categorical ({len(categorical_features)}): {categorical_features}")

print("\n1.5 Target Variable:")
print("Target: 'Churn' (Binary Categorical)")
print(f"Distribution:\n{df['Churn'].value_counts()}")
print(f"Percentage:\n{df['Churn'].value_counts(normalize=True) * 100}")

print("\n1.6 Overall Statistics:")
print(df.describe())

print("\n1.7 Categorical Statistics:")
print(df.describe(include=['object']))

print("\n1.8 Statistics by Churn Status:")
print("\nChurned Customers:")
print(df[df['Churn'] == 'Yes'].describe())
print("\nNon-Churned Customers:")
print(df[df['Churn'] == 'No'].describe())

#Compare means by churn status
print("\n1.9 Mean Comparison:")
comparison = pd.DataFrame({
    'Churned': df[df['Churn'] == 'Yes'][numerical_features].mean(),
    'Not Churned': df[df['Churn'] == 'No'][numerical_features].mean()
})
print(comparison)

print("\n" + "="*80 + "\n")


#STEP 2: MISSING DATA ANALYSIS

print("="*30)
print("STEP 2: MISSING DATA ANALYSIS")
print("="*30)

print("\n2.1 Missing Values Count:")
missing_values = df.isnull().sum()
print(missing_values)

print("\n2.2 Missing Values Percentage:")
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Count': missing_values, 'Percentage': missing_percentage})
print(missing_df[missing_df['Count'] > 0])

#Visualize missing data
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.tight_layout()
plt.savefig('missing_data_heatmap.png', dpi=300, bbox_inches='tight')
print("\n[Saved: missing_data_heatmap.png]")
plt.close()

#Checking TotalCharges data type
print(f"\n2.3 TotalCharges type: {df['TotalCharges'].dtype}")
if df['TotalCharges'].dtype == 'object':
    print(f"Non-numeric TotalCharges: {df[df['TotalCharges'] == ' '].shape[0]}")

    # Converting to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print(f"After conversion, missing: {df['TotalCharges'].isnull().sum()}")

    #Handling missing values: 0 for new customers, median for others
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
    if df['TotalCharges'].isnull().sum() > 0:
        median_charges = df['TotalCharges'].median()
        df['TotalCharges'].fillna(median_charges, inplace=True)
        print(f"Imputed with median: {median_charges:.2f}")

print(f"\n2.4 Final missing values: {df.isnull().sum().sum()}")
print("\n" + "="*80 + "\n")


#STEP 3: FEATURE DISTRIBUTION & OUTLIER ANALYSIS

print("="*30)
print("STEP 3: FEATURE DISTRIBUTION & OUTLIER ANALYSIS")
print("="*30)

print("\n3.1 Categorical Feature Distributions:")

#Bar plots for categorical features
cat_features_to_plot = [col for col in categorical_features if col not in ['customerID', 'Churn']]
n_cat = len(cat_features_to_plot)
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.ravel()

for idx, col in enumerate(cat_features_to_plot):
    df[col].value_counts().plot(kind='bar', ax=axes[idx], color='skyblue', edgecolor='black')
    axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count')
    axes[idx].tick_params(axis='x', rotation=45)

for idx in range(n_cat, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('categorical_distributions.png', dpi=300, bbox_inches='tight')
print("[Saved: categorical_distributions.png]")
plt.close()

print("\n3.2 Numerical Feature Distributions:")

#Box plots for numerical features
num_features_to_plot = [col for col in numerical_features if col != 'customerID']
fig, axes = plt.subplots(1, len(num_features_to_plot), figsize=(18, 5))

for idx, col in enumerate(num_features_to_plot):
    axes[idx].boxplot(df[col].dropna(), vert=True)
    axes[idx].set_title(f'{col} Box Plot', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(col)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('numerical_boxplots.png', dpi=300, bbox_inches='tight')
print("[Saved: numerical_boxplots.png]")
plt.close()

#Histograms for numerical features
fig, axes = plt.subplots(1, len(num_features_to_plot), figsize=(18, 5))

for idx, col in enumerate(num_features_to_plot):
    axes[idx].hist(df[col].dropna(), bins=30, color='lightcoral', edgecolor='black')
    axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('numerical_histograms.png', dpi=300, bbox_inches='tight')
print("[Saved: numerical_histograms.png]")
plt.close()

print("\n3.3 Outlier Detection (IQR Method):")

def detect_outliers_iqr(data, column):
    """Detect outliers using Interquartile Range method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

outlier_summary = []
for col in num_features_to_plot:
    outliers, lower, upper = detect_outliers_iqr(df, col)
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(df)) * 100

    outlier_summary.append({
        'Feature': col,
        'Outlier Count': outlier_count,
        'Percentage': f'{outlier_pct:.2f}%',
        'Lower Bound': f'{lower:.2f}',
        'Upper Bound': f'{upper:.2f}'
    })

    print(f"\n{col}: {outlier_count} outliers ({outlier_pct:.2f}%)")
    print(f"  Bounds: [{lower:.2f}, {upper:.2f}]")

print("\n3.4 Outlier Summary:")
print(pd.DataFrame(outlier_summary).to_string(index=False))

print("\n3.5 Decision: Keep all outliers (legitimate business values)")
print("\n" + "="*80 + "\n")


# STEP 4: DATA PRE-PROCESSING

print("="*30)
print("STEP 4: DATA PRE-PROCESSING")
print("="*30)

#Creating copy for preprocessing
df_processed = df.copy()

print("\n4.1 Remove customerID")
df_processed = df_processed.drop('customerID', axis=1)
print(f"Columns after removal: {df_processed.shape[1]}")

print("\n4.2 Encode Target Variable")
y = df_processed['Churn'].map({'No': 0, 'Yes': 1})
df_processed = df_processed.drop('Churn', axis=1)
print(f"Target encoded: No=0, Yes=1")
print(f"Distribution:\n{y.value_counts()}")

print("\n4.3 Encode Categorical Features")
#Identifying binary vs multi-class categoricals
binary_cols = []
multiclass_cols = []

for col in df_processed.select_dtypes(include=['object']).columns:
    if df_processed[col].nunique() == 2:
        binary_cols.append(col)
    else:
        multiclass_cols.append(col)

print(f"Binary columns ({len(binary_cols)}): {binary_cols}")
print(f"Multi-class columns ({len(multiclass_cols)}): {multiclass_cols}")

#Label encode binary columns
label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# One-hot encode multi-class columns
print("\n4.4 One-Hot Encoding:")
df_processed = pd.get_dummies(df_processed, columns=multiclass_cols, drop_first=True)
print(f"Shape after encoding: {df_processed.shape}")

print("\n4.5 Feature Scaling (StandardScaler)")
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
scaler = StandardScaler()
df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])
print(f"Scaled features: {cols_to_scale}")
print("\nSample scaled data:")
print(df_processed[cols_to_scale].head())

#Creating feature matrix
X = df_processed

print(f"\n4.6 Final Dataset Shape: {X.shape}")
print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
print("\n" + "="*80 + "\n")


#STEP 5: TARGET VARIABLE DISTRIBUTION

print("="*30)
print("STEP 5: TARGET VARIABLE DISTRIBUTION")
print("="*30)

print("\n5.1 Churn Distribution:")
churn_counts = y.value_counts()
churn_percentages = y.value_counts(normalize=True) * 100

print(f"No Churn (0): {churn_counts[0]} ({churn_percentages[0]:.2f}%)")
print(f"Churn (1): {churn_counts[1]} ({churn_percentages[1]:.2f}%)")

#Visualizing distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#Bar plot
y.value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'], edgecolor='black')
axes[0].set_title('Churn Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Churn (0=No, 1=Yes)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['No Churn', 'Churn'], rotation=0)
axes[0].grid(True, alpha=0.3)

for i, v in enumerate(y.value_counts().values):
    axes[0].text(i, v + 100, str(v), ha='center', va='bottom', fontweight='bold')

#Pie chart
axes[1].pie(y.value_counts().values, labels=['No Churn', 'Churn'],
           autopct='%1.1f%%', colors=['green', 'red'], startangle=90,
           explode=(0, 0.1))
axes[1].set_title('Churn Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
print("\n[Saved: target_distribution.png]")
plt.close()

print("\n5.2 Class Balance Analysis:")
minority_class = churn_counts.min()
majority_class = churn_counts.max()
imbalance_ratio = majority_class / minority_class

print(f"Majority class: {majority_class}")
print(f"Minority class: {minority_class}")
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 1.5:
    print("\nDataset is IMBALANCED")
    print("This requires special handling (class_weight='balanced', F1-score focus)")
else:
    print("\nDataset is BALANCED")

print("\n" + "="*80 + "\n")


#STEP 6: ML ALGORITHM SELECTION & TRAINING

print("="*30)
print("STEP 6: ML ALGORITHM SELECTION & TRAINING")
print("="*30)

print("\n6.1 Train-Test Split (80-20, stratified):")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"Features: {X_train.shape[1]}")

print("\nTrain class distribution:")
print(y_train.value_counts())
print(f"Percentages: {y_train.value_counts(normalize=True) * 100}")

print("\nTest class distribution:")
print(y_test.value_counts())
print(f"Percentages: {y_test.value_counts(normalize=True) * 100}")

# Calculate sample weights for Gradient Boosting (to handle class imbalance)
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
print(f"\n6.1.5 Sample Weights for Gradient Boosting:")
print(f"Weight for class 0 (No Churn): {sample_weights[y_train == 0][0]:.4f}")
print(f"Weight for class 1 (Churn): {sample_weights[y_train == 1][0]:.4f}")

print("\n6.2 Selected Algorithms:")
print("1. Logistic Regression - Baseline linear model")
print("2. Decision Tree - Interpretable non-linear model")
print("3. Random Forest - Ensemble method")
print("4. Gradient Boosting - Advanced sequential ensemble")
print("5. SVM - Margin-based classifier")

print("\n6.3 Training Models:")

#Define models
models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5, class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight='balanced', random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        min_samples_split=10, min_samples_leaf=4, subsample=0.8,
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf', class_weight='balanced', probability=True, random_state=42
    )
}

#Train all models
print("\n6.3 Training Models with Initial Parameters:")
print("(We'll tune some models later to improve performance)\n")

trained_models = {}
for name, model in models.items():
    print(f"Training {name}...", end=" ")
    if name == 'Gradient Boosting':
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
    trained_models[name] = model
    print("✓")

print("\n✓ All models trained successfully!")
print("\n" + "="*30 + "\n")


#STEP 6.5: HYPERPARAMETER TUNING

print("="*30)
print("STEP 6.5: HYPERPARAMETER TUNING")
print("="*30)

print("\nTrying to find better parameters for our models...")
print("This might take a minute...\n")

# Define parameter grids - trying a few different values for each parameter
# Random Forest - we'll try different numbers of trees and depths
param_grid_rf = {
    'n_estimators': [50, 100, 150],  # How many trees to use
    'max_depth': [5, 10, 15]  # How deep each tree can grow
}

# Decision Tree - we'll try different depths
param_grid_dt = {
    'max_depth': [3, 5, 7, 10]  # Different tree depths to try
}

# Tune Random Forest
print("6.5.1 Tuning Random Forest...")
print("Trying different combinations of n_estimators and max_depth...")

# GridSearchCV tries different combinations of parameters
# cv=3 means 3-fold cross-validation
# scoring='recall' focuses on catching churners (our main goal)
rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
grid_rf = GridSearchCV(rf_base, param_grid_rf, cv=3, scoring='recall', verbose=1)
grid_rf.fit(X_train, y_train)

print(f"\nBest parameters found: {grid_rf.best_params_}")
print(f"Best cross-validation recall score: {grid_rf.best_score_:.3f}")

# Tune Decision Tree
print("\n6.5.2 Tuning Decision Tree...")
print("Trying different max_depth values...")

dt_base = DecisionTreeClassifier(class_weight='balanced', random_state=42)
grid_dt = GridSearchCV(dt_base, param_grid_dt, cv=3, scoring='recall', verbose=1)
grid_dt.fit(X_train, y_train)

print(f"\nBest parameters found: {grid_dt.best_params_}")
print(f"Best cross-validation recall score: {grid_dt.best_score_:.3f}")

# Update our trained models with the tuned versions
# This replaces the old models with the improved ones
trained_models['Random Forest'] = grid_rf.best_estimator_
trained_models['Decision Tree'] = grid_dt.best_estimator_

# Tune Gradient Boosting
print("\n6.5.3 Tuning Gradient Boosting...")
print("Trying different combinations of n_estimators, learning_rate, and max_depth...")

param_grid_gb = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [4, 5, 6],
    'subsample': [0.8, 0.9]
}

gb_base = GradientBoostingClassifier(random_state=42)
grid_gb = GridSearchCV(gb_base, param_grid_gb, cv=3, scoring='recall', verbose=1)
grid_gb.fit(X_train, y_train, sample_weight=sample_weights)

print(f"\nBest parameters found: {grid_gb.best_params_}")
print(f"Best cross-validation recall score: {grid_gb.best_score_:.3f}")

trained_models['Gradient Boosting'] = grid_gb.best_estimator_

print("\n✓ Hyperparameter tuning completed!")
print("Updated Random Forest, Decision Tree, and Gradient Boosting with better parameters")
print("\n" + "="*80 + "\n")


#STEP 7: MODEL EVALUATION & RESULTS

print("="*30)
print("STEP 7: MODEL EVALUATION & RESULTS")
print("="*30)

print("\n7.1 Evaluation Metrics:")
print("- Accuracy: Overall correctness")
print("- Precision: Of predicted churners, how many actually churned?")
print("- Recall: Of actual churners, how many did we identify?")
print("- F1-Score: Harmonic mean of Precision and Recall")
print("- ROC-AUC: Area under ROC curve")

#Evaluate models
results = []
predictions = {}
probabilities = {}

print("\n7.2 Model Performance:\n")

for name, model in trained_models.items():
    #Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    predictions[name] = y_pred
    probabilities[name] = y_prob

    #Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        'Model': name,
        'Accuracy': f'{accuracy:.4f}',
        'Precision': f'{precision:.4f}',
        'Recall': f'{recall:.4f}',
        'F1-Score': f'{f1:.4f}',
        'ROC-AUC': f'{roc_auc:.4f}'
    })

    print(f"{name}:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}\n")

#Results table
results_df = pd.DataFrame(results)
print("7.3 Comparison Table:")
print(results_df.to_string(index=False))

#Comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    values = [float(results_df[results_df['Model'] == model][metric].values[0])
              for model in results_df['Model']]
    bars = ax.bar(results_df['Model'], values, color=colors[idx], edgecolor='black')
    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n[Saved: model_comparison.png]")
plt.close()

print("\n7.4 Confusion Matrices:")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for idx, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[idx]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                cbar_kws={'label': 'Count'})
    ax.set_title(f'{name} - Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

    print(f"\n{name}:")
    print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")

# Hide the empty 6th subplot
axes[5].axis('off')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("\n[Saved: confusion_matrices.png]")
plt.close()

print("\n7.5 ROC Curves:")

plt.figure(figsize=(10, 8))

for name, y_prob in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("\n[Saved: roc_curves.png]")
plt.close()

print("\n7.6 Best Model:")
best_f1_idx = results_df['F1-Score'].astype(float).idxmax()
best_model = results_df.loc[best_f1_idx, 'Model']
print(f"Based on F1-Score: {best_model}")

print("\n" + "="*30 + "\n")



