import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  # for saving model

# Paths
CSV_PATH = "output/pose_data.csv"
MODEL_PATH = "output/pose_classifier.pkl"

# Load dataset
df = pd.read_csv(CSV_PATH)

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
print("Training model...")
clf.fit(X_train, y_train)

# Evaluate on test set
y_pred = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
