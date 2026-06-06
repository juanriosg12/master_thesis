# Method-level group metrics — linear_conf_f50_s1000_p30

Scalar summaries behind the method-level scatter plots, using the standardised metric names: **Magnitude Divergence ΔM** (column `GSS_pct`/`TGA_pct`) = mean over features of the per-feature RMS(|φ_subject|−|φ_ref|)/model-output std, in %; **Sign Disagreement D** (column `SignDisagree_pct`) = mean over features of the % of instances whose sign disagrees with the reference. Reference subscripts: disc = PC↔LiNGAM, oracle = True/consensus DAG, base = Traditional.

## Cross-discovery instability — ΔM_disc vs D_disc (per Shapley method)
`GSS_pct` = ΔM_disc = RMS_i(|φ(PC)|−|φ(LiNGAM)|)/model-output std per feature × 100, averaged over features; `SSS_pct` = % of instances with matching sign PC vs LiNGAM; `SignDisagree_pct` = D_disc = 100 − SSS (the scatter's y-axis).

| Method | GSS_pct | SSS_pct | SignDisagree_pct |
| --- | --- | --- | --- |
| Asymmetric | 1.07 % | 94.07 % | 5.93 % |
| Causal | 7.15 % | 62.06 % | 37.94 % |
| Flow | 6.06 % | 86.73 % | 13.27 % |


## Alignment to True — ΔM_oracle vs D_oracle (per Shapley method × discovered graph)
`TGA_pct` = ΔM_oracle = RMS_i(|φ(disc)|−|φ(True)|)/model-output std per feature × 100, averaged over features; `SignAlign_pct` = % of instances with matching sign vs True; `SignDisagree_pct` = D_oracle = 100 − SignAlign (the scatter's y-axis).

| Method | Graph | Reference | TGA_pct | SignAlign_pct | SignDisagree_pct |
| --- | --- | --- | --- | --- | --- |
| Asymmetric | PC | True | 1.02 % | 94.12 % | 5.88 % |
| Asymmetric | LiNGAM | True | 0.79 % | 93.70 % | 6.30 % |
| Causal | PC | True | 5.32 % | 68.40 % | 31.60 % |
| Causal | LiNGAM | True | 7.57 % | 62.54 % | 37.46 % |
| Flow | PC | True | 7.34 % | 57.74 % | 42.26 % |
| Flow | LiNGAM | True | 8.07 % | 55.09 % | 44.91 % |


## Alignment to Traditional — ΔM_base vs D_base (per Shapley method × discovered graph)
`TGA_pct` = ΔM_base = RMS_i(|φ(disc)|−|φ(Traditional)|)/model-output std per feature × 100, averaged over features; `SignAlign_pct` = % of instances with matching sign vs Traditional; `SignDisagree_pct` = D_base = 100 − SignAlign (the scatter's y-axis).

| Method | Graph | Reference | TGA_pct | SignAlign_pct | SignDisagree_pct |
| --- | --- | --- | --- | --- | --- |
| Asymmetric | PC | Traditional | 0.68 % | 94.92 % | 5.08 % |
| Asymmetric | LiNGAM | Traditional | 0.86 % | 94.24 % | 5.76 % |
| Causal | PC | Traditional | 6.39 % | 64.70 % | 35.30 % |
| Causal | LiNGAM | Traditional | 5.84 % | 66.08 % | 33.92 % |
| Flow | PC | Traditional | 6.62 % | 63.42 % | 36.58 % |
| Flow | LiNGAM | Traditional | 6.61 % | 62.32 % | 37.68 % |
