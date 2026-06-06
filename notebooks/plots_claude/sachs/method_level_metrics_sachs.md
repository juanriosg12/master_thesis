# Method-level group metrics — sachs

Scalar summaries behind the method-level scatter plots. Magnitude metrics (GSS, TGA) are the mean over features of the per-feature RMS / output_range metric; sign metrics (SSS, Sign Alignment) are the mean over features.

## Graph-discovery instability — GSS vs SSS (per Shapley method)
`GSS` = RMS_i(|φ(PC)|−|φ(LiNGAM)|)/output_range per feature, averaged over features; `SSS` = sign agreement PC vs LiNGAM; `SignDisagree` = 1 − SSS (the scatter's y-axis).

| Method | GSS | SSS | SignDisagree |
| --- | --- | --- | --- |
| Asymmetric | 0.0089 | 0.8140 | 0.1860 |
| Causal | 0.0106 | 0.6820 | 0.3180 |
| Flow | 0.0157 | 0.7812 | 0.2188 |


## Alignment to True — TGA vs Sign Alignment (per Shapley method × discovered graph)
`TGA` = RMS_i(|φ(disc)|−|φ(True)|)/output_range per feature, averaged over features; `SignAlign` = sign agreement vs True; `SignDisagree` = 1 − SignAlign (the scatter's y-axis).

| Method | Graph | Reference | TGA | SignAlign | SignDisagree |
| --- | --- | --- | --- | --- | --- |
| Asymmetric | PC | True | 0.0096 | 0.7580 | 0.2420 |
| Asymmetric | LiNGAM | True | 0.0030 | 0.7660 | 0.2340 |
| Causal | PC | True | 0.0095 | 0.6370 | 0.3630 |
| Causal | LiNGAM | True | 0.0087 | 0.6770 | 0.3230 |
| Flow | PC | True | 0.0163 | 0.7164 | 0.2836 |
| Flow | LiNGAM | True | 0.0085 | 0.7619 | 0.2381 |


## Alignment to Traditional — TGA vs Sign Alignment (per Shapley method × discovered graph)
`TGA` = RMS_i(|φ(disc)|−|φ(Traditional)|)/output_range per feature, averaged over features; `SignAlign` = sign agreement vs Traditional; `SignDisagree` = 1 − SignAlign (the scatter's y-axis).

| Method | Graph | Reference | TGA | SignAlign | SignDisagree |
| --- | --- | --- | --- | --- | --- |
| Asymmetric | PC | Traditional | 0.0053 | 0.8320 | 0.1680 |
| Asymmetric | LiNGAM | Traditional | 0.0046 | 0.8300 | 0.1700 |
| Causal | PC | Traditional | 0.0082 | 0.6870 | 0.3130 |
| Causal | LiNGAM | Traditional | 0.0098 | 0.6690 | 0.3310 |
| Flow | PC | Traditional | 0.0133 | 0.5980 | 0.4020 |
| Flow | LiNGAM | Traditional | 0.0149 | 0.5850 | 0.4150 |
