# ✅ Dataset create
# ✅ Dataset load
# ✅ Categorical data → numbers
# ✅ X & y split
# ✅ Train/Test Split
# ✅ Feature Scaling
# ✅ Logistic Regression model
# ✅ Prediction
# ✅ Accuracy + Confusion Matrix + Classification Report
# ✅ New Customer Churn Prediction
import pandas as pd
# LabelEncoder is used to convert text/categorical values into numbers
# For example: # Contract,month-to-month,one year,two year
# Machine Learning model-ku text direct-ah understand panna mudiyadhu.
# So it converts them into numbers, for example:
# Month-to-month → 0
# One year       → 1
# Two year       → 2
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("customer churn prediction.csv")

# Display dataset
print(data)

# Convert categorical data into numbers
encoder = LabelEncoder()
# LabelEncoder converts them into numbers[for contract]
# fit → learns the different categories.
# transform → converts those categories into numbers
# fit transform means first learn the value and convert into numbers
data["Contract"] = encoder.fit_transform(data["Contract"])
data["InternetService"] = encoder.fit_transform(data["InternetService"])
# churn for yes or no
data["Churn"] = encoder.fit_transform(data["Churn"])

# Separate input and output
# axis 1 for remove chrun column
X = data.drop("Churn", axis=1)
y = data["Churn"]

# step 4

# Display X
print("\nX - Input Features:")
print(X)

# Display y
print("\ny - Target:")
print(y)

# step 5
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# Dataset split pannumbodhu random-ah rows select aagum 
# Random split same-ah repeat aaganum na random_state=42 use panrom

print("X_train:")
print(X_train)

print("\nX_test:")
print(X_test)

print("\ny_train:")
print(y_train)

print("\ny_test:")
print(y_test)

# step 6
from sklearn.preprocessing import StandardScaler
# StandardScaler is used for feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("Scaled X_train:")
print(X_train)

print("\nScaled X_test:")
print(X_test)

# step 7
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

print("Logistic Regression model trained successfully!")

model = LogisticRegression()
model.fit(X_train, y_train)

# step 8
y_pred = model.predict(X_test)

print("Actual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

# step 9
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:")
print(accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# step 10
# Step 10: Predict churn for a new customer

new_customer = pd.DataFrame({
    "Age": [30],
    "MonthlyCharges": [70],
    "Tenure": [12],
    "Contract": [0],        # 0 = Month-to-month
    "InternetService": [1]  # 1 = Fiber
})

# Scale the new customer data
new_customer_scaled = scaler.transform(new_customer)

# Make prediction
prediction = model.predict(new_customer_scaled)

print("\nNew Customer Prediction:")

if prediction[0] == 1:
    print("Customer is likely to Churn")
else:
    print("Customer is likely to Stay")