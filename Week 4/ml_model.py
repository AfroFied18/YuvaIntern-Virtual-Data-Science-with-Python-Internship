import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, roc_auc_score, classification_report)

sns.set_theme(style="whitegrid")
PALETTE = ["#2E5EAA", "#E8703A", "#3FA796", "#C74B50", "#8E6BAF"]
np.random.seed(42)

# ============================================================
# 1. LOAD DATA & DEFINE TARGET
# ============================================================
df = pd.read_csv("retail_sales_synthetic.csv")
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["MonthNum"] = df["OrderDate"].dt.month

# Target: was this order profitable? (this is the natural next question
# after Week 3, where we found discount level strongly predicts margin)
df["IsProfitable"] = (df["Profit"] > 0).astype(int)

print("Class balance:")
print(df["IsProfitable"].value_counts(normalize=True))

# ============================================================
# 2. DATA PREPARATION
# ============================================================
# Features deliberately EXCLUDE Sales/Profit/ProfitMargin themselves
# (that would be leaking the answer). We only use info that would be
# known at order time: discount given, quantity, category, region,
# segment, and month.
feature_cols_num = ["Discount", "Quantity", "MonthNum"]
feature_cols_cat = ["Category", "Region", "Segment"]

X = pd.get_dummies(df[feature_cols_num + feature_cols_cat], columns=feature_cols_cat, drop_first=True)
y = df["IsProfitable"]

print("\nMissing values per column:\n", df[feature_cols_num + feature_cols_cat].isna().sum())
print("\nFinal feature matrix shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# scale numeric features for logistic regression (tree doesn't need it,
# but scaling the same split keeps things simple and doesn't hurt trees)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[feature_cols_num] = scaler.fit_transform(X_train[feature_cols_num])
X_test_scaled[feature_cols_num] = scaler.transform(X_test[feature_cols_num])

# ============================================================
# 3. MODEL 1: LOGISTIC REGRESSION
# ============================================================
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred_lr = logreg.predict(X_test_scaled)
y_prob_lr = logreg.predict_proba(X_test_scaled)[:, 1]

# ============================================================
# 4. MODEL 2: DECISION TREE (comparison model)
# ============================================================
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=42)
tree.fit(X_train, y_train)  # trees don't need scaling
y_pred_tree = tree.predict(X_test)
y_prob_tree = tree.predict_proba(X_test)[:, 1]

# ============================================================
# 5. METRICS
# ============================================================
def report(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Not Profitable", "Profitable"]))
    return dict(model=name, accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc)

results_lr = report("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
results_tree = report("Decision Tree (depth=4)", y_test, y_pred_tree, y_prob_tree)

results_df = pd.DataFrame([results_lr, results_tree])
print("\nSummary:\n", results_df)

# ============================================================
# 6. VISUALIZATIONS
# ============================================================

# --- Chart 1: Confusion matrix, Logistic Regression ---
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6.5, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Profitable", "Profitable"],
            yticklabels=["Not Profitable", "Profitable"], cbar=False, annot_kws={"size": 16})
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix \u2013 Logistic Regression")
plt.tight_layout()
plt.savefig("charts/m1_confusion_matrix.png", dpi=150)
plt.close()

# --- Chart 2: ROC curves, both models ---
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)

plt.figure(figsize=(7.5, 6.5))
plt.plot(fpr_lr, tpr_lr, color=PALETTE[0], linewidth=2.5,
         label=f"Logistic Regression (AUC = {results_lr['auc']:.3f})")
plt.plot(fpr_tree, tpr_tree, color=PALETTE[1], linewidth=2.5,
         label=f"Decision Tree (AUC = {results_tree['auc']:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("charts/m2_roc_curve.png", dpi=150)
plt.close()

# --- Chart 3: Metric comparison bar chart ---
metrics = ["accuracy", "precision", "recall", "f1", "auc"]
x = np.arange(len(metrics))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, [results_lr[m] for m in metrics], width, label="Logistic Regression", color=PALETTE[0])
plt.bar(x + width/2, [results_tree[m] for m in metrics], width, label="Decision Tree", color=PALETTE[1])
plt.xticks(x, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("charts/m3_metric_comparison.png", dpi=150)
plt.close()

# --- Chart 4: Logistic regression coefficients (feature importance / direction) ---
coefs = pd.Series(logreg.coef_[0], index=X_train_scaled.columns).sort_values()
plt.figure(figsize=(9, 7))
colors = [PALETTE[1] if c < 0 else PALETTE[0] for c in coefs.values]
plt.barh(coefs.index, coefs.values, color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Coefficient (standardized)")
plt.title("Logistic Regression Coefficients\n(negative = pushes toward 'Not Profitable')")
plt.tight_layout()
plt.savefig("charts/m4_coefficients.png", dpi=150)
plt.close()

print("\nAll charts saved to charts/")
