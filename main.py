import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import joblib

df = pd.read_csv('Crop_Dataset.csv')

print(df.shape)  # added this to check the size of the dataset.

df = df.dropna(axis='index', how='any')  # any row which has a null value will be dropped out.

label_encoder = LabelEncoder()
df['Label_Encoded'] = label_encoder.fit_transform(df['Label'])

scaler = StandardScaler()
features_to_scale = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

df.to_csv('new_dataset.csv')
X = df[features_to_scale]
y = df['Label_Encoded']

model = svm.SVC()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)

print("Model's accuracy is: " + str(acc))
joblib.dump(model, 'model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')  # Save the LabelEncoder

N = float(input("Enter the level of Nitrogen (N): "))
P = float(input("Enter the level of Phosphorus (P): "))
K = float(input("Enter the level of Potassium (K): "))
temperature = float(input("Enter the temperature: "))
humidity = float(input("Enter the humidity: "))
ph = float(input("Enter the pH value: "))
rainfall = float(input("Enter the rainfall: "))

input_df = pd.DataFrame({
    "N": [N],
    "P": [P],
    "K": [K],
    "temperature": [temperature],
    "humidity": [humidity],
    "ph": [ph],
    "rainfall": [rainfall]
})

input_df[features_to_scale] = scaler.transform(input_df[features_to_scale])  # Use transform, not fit_transform

model = joblib.load('model.joblib')
label_encoder = joblib.load('label_encoder.joblib')  # Load the LabelEncoder

decision_scores = model.decision_function(input_df)

# Convert decision function scores to probabilities using softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

predicted_probabilities = softmax(decision_scores)

# Get the top 3 predictions for each sample
top_three_indices = np.argsort(predicted_probabilities, axis=1)[:, -3:]

crop_names = label_encoder.inverse_transform(model.classes_)

# Print the names of top three crops for each sample
for i, indices in enumerate(top_three_indices):
    top_three_crops = [crop_names[index] for index in indices]
    print(f"Sample {i+1}: Top Three Crops - {top_three_crops}")
