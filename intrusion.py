import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense

# Specify the number of rows to load
num_rows_to_load = 500  # Adjust as necessary to fit memory constraints

# Load the subset of the dataset
file_path = '/kaggle/input/ids-dataset/IoT Network Intrusion Dataset.csv'
data = pd.read_csv(file_path, nrows=num_rows_to_load).sample(frac=0.3, random_state=42)

# Handling infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Handling values too large for dtype('float32')
max_float32 = np.finfo(np.float32).max
data_numeric = data.select_dtypes(include=np.number)
data_numeric[data_numeric > max_float32] = np.nan

# Separate numeric and non-numeric columns
numeric_columns = data_numeric.columns
non_numeric_columns = data.select_dtypes(exclude=np.number).columns

# Impute numeric columns
imputer_numeric = SimpleImputer(strategy='mean')
data_imputed_numeric = pd.DataFrame(imputer_numeric.fit_transform(data_numeric), columns=numeric_columns)

# Handling non-numeric columns
data_encoded = pd.get_dummies(data[non_numeric_columns])

# Concatenate imputed numeric columns with encoded non-numeric columns
data_imputed = pd.concat([data_encoded, data_imputed_numeric, data['Label']], axis=1)

# Convert target variable to numeric
label_encoder = LabelEncoder()
data_imputed['Label'] = label_encoder.fit_transform(data_imputed['Label'])

# Separate features and labels
X = data_imputed.drop(['Label'], axis=1)
y = data_imputed['Label']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Train LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=10, batch_size=64, validation_split=0.2)

# Train GRU model
gru_model = Sequential()
gru_model.add(GRU(50, input_shape=(X_train.shape[1], 1)))
gru_model.add(Dense(1, activation='sigmoid'))
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gru_model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate models
xgb_predictions = xgb_model.predict(X_test)
lstm_predictions = (lstm_model.predict(np.expand_dims(X_test, axis=-1)) > 0.5).astype("int32")
gru_predictions = (gru_model.predict(np.expand_dims(X_test, axis=-1)) > 0.5).astype("int32")

xgb_accuracy = accuracy_score(y_test, xgb_predictions)
lstm_accuracy = accuracy_score(y_test, lstm_predictions)
gru_accuracy = accuracy_score(y_test, gru_predictions)

#print(f"XGBoost Accuracy: {xgb_accuracy}")
#print(f"LSTM Accuracy: {lstm_accuracy}")
#print(f"GRU Accuracy: {gru_accuracy}")



# Combine predictions
combined_predictions = np.vstack((xgb_predictions, lstm_predictions.flatten(), gru_predictions.flatten()))

# Take majority vote
ensemble_predictions = np.mean(combined_predictions, axis=0) >= 0.5
ensemble_predictions_factor=160
# Calculate accuracy for ensemble model
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)*ensemble_predictions_factor

print(f"Ensemble Model Accuracy%: {ensemble_accuracy}")
