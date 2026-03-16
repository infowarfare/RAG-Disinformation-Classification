import numpy as np
from sklearn.metrics import f1_score
from statsmodels.stats.multitest import multipletests

# Test datasets
y_true  = [1,0,1,1,0,1,0,1,0,1]
pred_a  = [1,0,1,1,0,1,0,1,1,1]
pred_b  = [1,0,0,1,0,1,0,0,0,1]

n_boot = 10000
diffs = []
n = len(y_true)

for _ in range(n_boot):
    
    # Bootstrap-Indices 
    idx = np.random.choice(n, n, replace=True)
    
    # Resampline of lists
    y_sample  = [y_true[i] for i in idx]
    a_sample  = [pred_a[i] for i in idx]
    b_sample  = [pred_b[i] for i in idx]
    
    f1_a = f1_score(y_sample, a_sample)
    f1_b = f1_score(y_sample, b_sample)
    
    # save diff delta
    diffs.append(f1_a - f1_b)

diffs = np.array(diffs)

print(f"F1 Score Model A: {f1_score(y_true, pred_a)}")
print(f"F1 Score Model B: {f1_score(y_true, pred_b)}")

# Simple diff
obs_diff = f1_score(y_true, pred_a) - f1_score(y_true, pred_b)

# calculated p-value
p_value = np.mean(diffs <= 0)

print("Observed diff:", obs_diff)
print("p-value:", p_value)

# Confidence interval
lower = np.percentile(diffs, 2.5)
upper = np.percentile(diffs, 97.5)

print("95% CI:", lower, upper)