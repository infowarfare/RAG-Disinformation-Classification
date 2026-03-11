import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Beispiel
X_val = ["I love this movie!", "The film was terrible.", "It was a great experience.", "Not worth watching."]
y_val = [1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# test predictions
predictions_model_1_original = np.array([1, 0, 1, 0])  # Vorhersagen Platzhalter
predictions_model_2_original = np.array([1, 0, 0, 0])  

# Calculate accuracy for both models
accuracy_model_1 = f1_score(y_val, predictions_model_1_original, average="macro")
accuracy_model_2 = f1_score(y_val, predictions_model_2_original, average="macro")

# --- Erweitert: p-Wert Berechnung ---
# Die wahre Beobachtungsdifferenz (wird für den p-Wert nicht direkt benötigt, aber zur Orientierung)
observed_diff = accuracy_model_1 - accuracy_model_2
# ------------------------------------

# Anzahl der Iterationen
n_iterations = 10000

# Differenzen
performance_diffs = []

# Resampling
for i in range(n_iterations):
    # Resampling mit Eliminierung
    indices = np.random.choice(len(y_val), size=len(y_val), replace=True)
    
    # Resampling der Indizes
    y_resampled = np.array([y_val[i] for i in indices])
    
    
    predictions_model_1_resampled = predictions_model_1_original[indices]
    predictions_model_2_resampled = predictions_model_2_original[indices]
    
    # F1 Score Berechnung
    accuracy_model_1_resampled = f1_score(y_resampled, predictions_model_1_resampled, average="macro")
    accuracy_model_2_resampled = f1_score(y_resampled, predictions_model_2_resampled, average="macro")

    # Berechnung der Differenz
    performance_diffs.append(accuracy_model_1_resampled - accuracy_model_2_resampled)

performance_diffs = np.array(performance_diffs)

# 95% Konfidenzintervall
mean_diff = np.mean(performance_diffs)
lower_bound = np.percentile(performance_diffs, 2.5)
upper_bound = np.percentile(performance_diffs, 97.5)

# --- NEU: p-Wert Berechnung ---
# Die Nullhypothese H0 lautet: Modell 1 ist NICHT besser als Modell 2 (Differenz <= 0).
# Der p-Wert ist der Anteil der Bootstrap-Differenzen, die die Nullhypothese stützen.
# Wir zählen, wie oft die Differenz 0 oder kleiner ist.
p_value = np.sum(performance_diffs <= 0) / n_iterations

# Ergebnisse
print(f"--- Modell-Performance auf Original-Daten ---")
print(f"Accuracy Model 1: {accuracy_model_1:.4f}")
print(f"Accuracy Model 2: {accuracy_model_2:.4f}")
print(f"Observed Difference (M1 - M2): {observed_diff:.4f}")
print("--- Bootstrap-Ergebnisse ---")
print(f"Bootstrap Mean Difference: {mean_diff:.4f}")
print(f"95% Confidence Interval: ({lower_bound:.4f}, {upper_bound:.4f})")
print(f"Einseitiger p-Wert (H0: M1 <= M2): {p_value:.4f}")

# Hypothesen
alpha = 0.05
if p_value < alpha:
    print(f"\nDa p-Wert ({p_value:.4f}) < alpha ({alpha}), können wir H0 verwerfen: **Modell 1 ist signifikant besser als Modell 2.**")
else:
    print(f"\nDa p-Wert ({p_value:.4f}) >= alpha ({alpha}), können wir H0 nicht verwerfen: **Es gibt keinen signifikanten Unterschied, dass Modell 1 besser ist.**")