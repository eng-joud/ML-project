import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv(r"Data\Original Data\Loan_Data.csv")
df.dropna(inplace=True)
df.drop(columns=["Loan_ID"], inplace=True)

le_loan = LabelEncoder()
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
df["Married"] = LabelEncoder().fit_transform(df["Married"])
df["Dependents"] = LabelEncoder().fit_transform(df["Dependents"])
df["Education"] = LabelEncoder().fit_transform(df["Education"])
df["Self_Employed"] = LabelEncoder().fit_transform(df["Self_Employed"])
df["Property_Area"] = LabelEncoder().fit_transform(df["Property_Area"])
df["Loan_Status"] = le_loan.fit_transform(df["Loan_Status"])

X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

features = X.columns.tolist()

scaler = StandardScaler()
scaler.fit(X_train[features])
X_train_scaled = scaler.transform(X_train[features])
X_test_scaled = scaler.transform(X_test[features])

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

os.makedirs("Data/Preprocessed data", exist_ok=True)
os.makedirs("Data/Results", exist_ok=True)

pd.DataFrame(X_train_balanced, columns=features).to_csv("Data/Preprocessed data/X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=features).to_csv("Data/Preprocessed data/X_test.csv", index=False)
y_train_balanced.to_csv("Data/Preprocessed data/Y_train.csv", index=False)
y_test.to_csv("Data/Preprocessed data/Y_test.csv", index=False)

models = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, class_weight='balanced'),
    "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "NaiveBayes": GaussianNB(),
    "ANN": MLPClassifier(max_iter=1000, random_state=42, early_stopping=True),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
}

loan_classes = ['Not Approved', 'Approved']

results = []

for name, model in models.items():
    print(f"\n== Training {name} ==")
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_scaled)

    pd.DataFrame({"Prediction": y_pred}).to_csv(f"Data/Results/prediction_{name}.csv", index=False)

    report = classification_report(y_test, y_pred, target_names=loan_classes, zero_division=0, output_dict=True)
    accuracy = report['accuracy']
    recall_class_0 = report['Not Approved']['recall']
    f1_class_0 = report['Not Approved']['f1-score']
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Recall_Class0": recall_class_0,
        "F1_Class0": f1_class_0
    })

    print(classification_report(y_test, y_pred, target_names=loan_classes, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=loan_classes,
                yticklabels=loan_classes)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"Data/Results/confusion_matrix_{name}.png")
    plt.close()

results_df = pd.DataFrame(results)

plt.figure(figsize=(12, 6))
bar_width = 0.25
x = range(len(results_df))
plt.bar([p - bar_width for p in x], results_df["Accuracy"], width=bar_width, label='Accuracy')
plt.bar(x, results_df["Recall_Class0"], width=bar_width, label='Recall (Class 0)')
plt.bar([p + bar_width for p in x], results_df["F1_Class0"], width=bar_width, label='F1-score (Class 0)')
plt.xticks(ticks=x, labels=results_df["Model"])
plt.ylabel("Score")
plt.title("Model Comparison After Applying SMOTE and class_weight")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Data/Results/model_comparison_barplot.png")
plt.show()
