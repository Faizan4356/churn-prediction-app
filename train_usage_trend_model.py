"""
train_usage_trend_model.py

Trains an LSTM to classify each customer's 5-month usage sequence as
"Declining", "Stable", or "Growing".

NOTE ON FRAMEWORK: the original spec called for TensorFlow/Keras, but
TensorFlow has no published build for Python 3.14 (this project's
interpreter) at the time of writing, so this uses PyTorch instead -
same architecture (a single-layer LSTM feeding a linear classifier),
same task, just a different framework. See usage_trend_model_def.py
for the model class (kept separate so app.py can import it too).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

from usage_trend_model_def import UsageTrendLSTM

torch.manual_seed(42)

usage_df = pd.read_csv("usage_history.csv")
usage_cols = [c for c in usage_df.columns if c.startswith("month_")]
sequences = usage_df[usage_cols].to_numpy(dtype=np.float32)  # (n, 5)

# ---------------------------------------------------------------
# 1. Label each sequence by trend: compare first-2-month average to
# last-2-month average, with a +/-5% band counted as "Stable".
# ---------------------------------------------------------------
first_two_avg = sequences[:, :2].mean(axis=1)
last_two_avg = sequences[:, -2:].mean(axis=1)
pct_change = (last_two_avg - first_two_avg) / first_two_avg

labels = np.where(pct_change < -0.05, "Declining",
          np.where(pct_change > 0.05, "Growing", "Stable"))

label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(labels)
print("Label distribution:", pd.Series(labels).value_counts().to_dict())

# ---------------------------------------------------------------
# 2. Normalize each sequence (per-row) so the LSTM learns shape/trend
# rather than absolute usage level, then reshape to (n, seq_len, 1).
# ---------------------------------------------------------------
seq_mean = sequences.mean(axis=1, keepdims=True)
seq_std = sequences.std(axis=1, keepdims=True) + 1e-6
sequences_norm = (sequences - seq_mean) / seq_std
X_all = sequences_norm.reshape(-1, 5, 1)

X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
    X_all, y_all, sequences, test_size=0.2, stratify=y_all, random_state=42
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# ---------------------------------------------------------------
# 3. Build and train the LSTM
# ---------------------------------------------------------------
model = UsageTrendLSTM(input_size=1, hidden_size=16, num_classes=len(label_encoder.classes_))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 60
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = loss_fn(logits, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {loss.item():.4f}")

# ---------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_preds = test_logits.argmax(dim=1).numpy()

lstm_acc = accuracy_score(y_test, test_preds)
print(f"\nLSTM test accuracy: {lstm_acc:.3f}")
print("Confusion matrix (rows=actual, cols=predicted), classes:", list(label_encoder.classes_))
print(confusion_matrix(y_test, test_preds))

# ---------------------------------------------------------------
# 5. Baseline for comparison: just check if the last value is lower
# than the first value -> "Declining", else "Growing" (this baseline
# can never predict "Stable", which is the point - it shows why a
# model that can weigh the whole trajectory is worth it).
# ---------------------------------------------------------------
baseline_labels = np.where(seq_test[:, -1] < seq_test[:, 0], "Declining", "Growing")
baseline_preds_encoded = np.array([label_encoder.transform([lab])[0] for lab in baseline_labels])
baseline_acc = accuracy_score(y_test, baseline_preds_encoded)

print(f"\nBaseline (first-vs-last value only) accuracy: {baseline_acc:.3f}")
print(f"LSTM accuracy:                                  {lstm_acc:.3f}")
if lstm_acc > baseline_acc:
    print(f"-> LSTM outperforms the naive baseline by {(lstm_acc - baseline_acc) * 100:.1f} points.")
else:
    print("-> LSTM did not beat the naive baseline on this run.")

# ---------------------------------------------------------------
# 6. Save model weights + label encoder
# ---------------------------------------------------------------
torch.save(model.state_dict(), "usage_trend_model.pt")
joblib.dump(label_encoder, "usage_trend_labels.joblib")
print("\nSaved usage_trend_model.pt and usage_trend_labels.joblib")
