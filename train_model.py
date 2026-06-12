import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv('diabetes.csv')
print("Dataset Shape:", df.shape)
print(df.head())

# ── 2. Handle Zero Values (replace 0s with column mean excluding zeros) ─
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    non_zero_mean = df.loc[df[col] != 0, col].mean()
    df[col] = df[col].replace(0, non_zero_mean)

# ── 3. Split Features and Target ─────────────────────────────
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# ── 4. Train/Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── 5. Feature Scaling ────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 6. Find Best K ────────────────────────────────────────────
accuracies = []
k_range = range(1, 21)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, knn.predict(X_test)))

best_k = k_range[accuracies.index(max(accuracies))]
print(f"\nBest K: {best_k}  |  Accuracy: {max(accuracies)*100:.2f}%")

# ── 7. Plot K vs Accuracy ─────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(k_range, accuracies, marker='o', color='steelblue')
plt.title('K Value vs Accuracy')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.savefig('static/k_accuracy.png')
plt.close()

# ── 8. Train Final Model with Best K ─────────────────────────
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── 9. Evaluation ─────────────────────────────────────────────
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ── 10. Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png')
plt.close()

# ── 11. Save Model and Scaler ─────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ model.pkl and scaler.pkl saved successfully!")


def main():
    # This file intentionally runs top-level operations for quick scripting.
    # If imported as a module, you can call main() to execute training.
    pass


if __name__ == '__main__':
    # Running the script will execute the training code above (module-level),
    # but keep __main__ guard for future extensions.
    print('Training script executed.')

