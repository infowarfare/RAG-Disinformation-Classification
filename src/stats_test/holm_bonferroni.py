from statsmodels.stats.multitest import multipletests

# Liste mit bereits berechneten p-Werten
pvals = [0.003, 0.01, 0.02, 0.04, 0.12, 0.20]

# Holm-Bonferroni-Korrektur
reject, pvals_corrected, _, _ = multipletests(
    pvals,
    alpha=0.05,
    method="holm"
)

# Ergebnisse anzeigen
for i, p in enumerate(pvals):
    print(
        f"Test {i+1}: raw p = {p:.5f}, "
        f"corrected p = {pvals_corrected[i]:.5f}, "
        f"significant = {reject[i]}"
    )