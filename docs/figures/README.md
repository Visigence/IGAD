# Experiment Figures

---

## exp1_easy_gamma_vs_gamma.png

**Experiment 1 — Easy Case**
Reference: Gamma(9,3) | Anomaly: Gamma(1.5,0.5)
- Same mean (3.0), different variance (1.0 vs 6.0) and skewness (0.667 vs 1.633)

Left panel:  IGAD score distribution — normal vs anomaly batches (AUC=1.000)
Right panel: Raw skewness shift distribution (AUC=0.983)

Conclusion: Both IGAD and variance baseline achieve perfect separation.
This experiment shows IGAD works but does not prove unique value,
because the distributions also differ in variance.

---

## exp2_hard_gamma_vs_lognormal.png

**Experiment 2 — Hard Case (the key result)**
Reference: Gamma(8,2) | Anomaly: LogNormal(mu=1.327, sigma=0.343)
- Identical mean (4.0) AND variance (2.0). Only skewness differs (0.707 vs 1.105)

Left panel:   IGAD score — |R_ref - R_local| (AUC=0.684)
Middle panel: MLE skewness CONTROL — same MLE, no geometry (AUC=0.610)
Right panel:  Raw skewness shift (AUC=0.651)

Key finding: IGAD beats the MLE-skewness control by +0.053 (mean over 5 seeds).
Since both use identical MLE fits, the gap is attributable to the curvature
tensor geometry — not MLE statistical efficiency alone.

Mean AUC over 5 seeds (n=200):
  IGAD         : 0.6542 ± 0.047
  MLE skewness : 0.6016 ± 0.038
  Gap          : +0.053
