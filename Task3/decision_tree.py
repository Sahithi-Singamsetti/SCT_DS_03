import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables

# Correct file path
file_path = "bank+marketing/bank-additional/bank-additional-full.csv"

# Load dataset
df = pd.read_csv(file_path, sep=';')

# Encode categorical variables
label_encoders = {}  # Store encoders for reference
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Save encoder if needed for later

# Define features (X) and target (y)
X = df.drop(columns=["y"])  # Assuming 'y' is the target column
y = df["y"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predictions and accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Decision Tree Model Accuracy: {accuracy:.2f}")
