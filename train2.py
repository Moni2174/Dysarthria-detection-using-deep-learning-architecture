import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout, BatchNormalization, \
    GlobalAveragePooling1D, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# 📂 Load extracted features
csv_file = r"C:\Users\monis\Documents\Samitra audio project\feature.csv"
df = pd.read_csv(csv_file)

# 🔹 Extract features and labels
X = df.iloc[:, 1:-1].values  # Exclude filename & label
y = df["Label"].values  # Target labels

# 🔹 Encode labels into numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 🔹 Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Reshape for CNN + LSTM
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# 🔹 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 🔹 Compute Class Weights (Balances Dysarthria & Non-Dysarthria)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"📊 Class Weights Applied: {class_weight_dict}")

# 🏗️ **Balanced CNN + BiLSTM Model**
model = Sequential([
    # 🔹 CNN Feature Extractor
    Conv1D(128, kernel_size=5, padding='same', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),

    Conv1D(256, kernel_size=3, padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),

    Conv1D(512, kernel_size=3, padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # 🔹 LSTM Sequence Learner
    Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2)),
    Dropout(0.3),

    Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.2)),
    Dropout(0.3),

    GlobalAveragePooling1D(),  # ✅ Moved here to avoid 2D shape issue

    # 🔹 Fully Connected Layers
    Dense(128),
    LeakyReLU(),
    Dropout(0.3),

    Dense(len(np.unique(y_encoded)), activation='softmax')  # Output Layer
])

# 🔹 Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),  # 🔹 Reduced LR for stability
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ⏳ **Callbacks**
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

# 🚀 **Train Model**
history = model.fit(
    X_train, y_train, epochs=25, batch_size=16, validation_data=(X_test, y_test),
    class_weight=class_weight_dict, callbacks=[early_stop, reduce_lr]
)

# 🏆 **Evaluate Model**
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 💾 Save Model
model.save("dysarthria_cnn_bilstm_balanced.h5")
print("✅ Model training complete! Saved as 'dysarthria_cnn_bilstm_balanced.h5'")

# 📊 **Training & Validation Plots**
plt.figure(figsize=(14, 6))

# 🔹 Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid()

# 🔹 Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid()

plt.show()

# 📉 **Confusion Matrix**
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
