# Method-level group metrics — sachs

Scalar summaries behind the method-level scatter plots, using the standardised metric names: **Magnitude Divergence ΔM** (column `GSS_pct`/`TGA_pct`) = mean over features of the per-feature RMS(|φ_subject|−|φ_ref|)/model-output std, in %; **Sign Disagreement D** (column `SignDisagree_pct`) = mean over features of the % of instances whose sign disagrees with the reference. Reference subscripts: disc = PC↔LiNGAM, oracle = True/consensus DAG, base = Traditional.

## Cross-discovery instability — ΔM_disc vs D_disc (per Shapley method)
`GSS_pct` = ΔM_disc = RMS_i(|φ(PC)|−|φ(LiNGAM)|)/model-output std per feature × 100, averaged over features; `SSS_pct` = % of instances with matching sign PC vs LiNGAM; `SignDisagree_pct` = D_disc = 100 − SSS (the scatter's y-axis).

| Method | GSS_pct | SSS_pct | SignDisagree_pct |
| --- | --- | --- | --- |
| Asymmetric | 8.46 % | 81.40 % | 18.60 % |
| Causal | 10.11 % | 68.20 % | 31.80 % |
| Flow | 14.92 % | 78.12 % | 21.88 % |


## Alignment to True — ΔM_oracle vs D_oracle (per Shapley method × discovered graph)
`TGA_pct` = ΔM_oracle = RMS_i(|φ(disc)|−|φ(True)|)/model-output std per feature × 100, averaged over features; `SignAlign_pct` = % of instances with matching sign vs True; `SignDisagree_pct` = D_oracle = 100 − SignAlign (the scatter's y-axis).

| Method | Graph | Reference | TGA_pct | SignAlign_pct | SignDisagree_pct |
| --- | --- | --- | --- | --- | --- |
| Asymmetric | PC | True | 9.17 % | 75.80 % | 24.20 % |
| Asymmetric | LiNGAM | True | 2.84 % | 76.60 % | 23.40 % |
| Causal | PC | True | 9.01 % | 63.70 % | 36.30 % |
| Causal | LiNGAM | True | 8.31 % | 67.70 % | 32.30 % |
| Flow | PC | True | 15.52 % | 71.64 % | 28.36 % |
| Flow | LiNGAM | True | 8.12 % | 76.19 % | 23.81 % |


## Alignment to Traditional — ΔM_base vs D_base (per Shapley method × discovered graph)
`TGA_pct` = ΔM_base = RMS_i(|φ(disc)|−|φ(Traditional)|)/model-output std per feature × 100, averaged over features; `SignAlign_pct` = % of instances with matching sign vs Traditional; `SignDisagree_pct` = D_base = 100 − SignAlign (the scatter's y-axis).

| Method | Graph | Reference | TGA_pct | SignAlign_pct | SignDisagree_pct |
| --- | --- | --- | --- | --- | --- |
| Asymmetric | PC | Traditional | 5.08 % | 83.20 % | 16.80 % |
| Asymmetric | LiNGAM | Traditional | 4.41 % | 83.00 % | 17.00 % |
| Causal | PC | Traditional | 7.83 % | 68.70 % | 31.30 % |
| Causal | LiNGAM | Traditional | 9.31 % | 66.90 % | 33.10 % |
| Flow | PC | Traditional | 12.68 % | 59.80 % | 40.20 % |
| Flow | LiNGAM | Traditional | 14.13 % | 58.50 % | 41.50 % |
