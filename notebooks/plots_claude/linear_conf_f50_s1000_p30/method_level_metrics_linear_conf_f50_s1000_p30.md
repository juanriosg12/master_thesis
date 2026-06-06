# Method-level group metrics — linear_conf_f50_s1000_p30

Scalar summaries behind the method-level scatter plots. Magnitude metrics (GSS, TGA) are the mean over features of the per-feature RMS / output_range metric; sign metrics (SSS, Sign Alignment) are the mean over features.

## Graph-discovery instability — GSS vs SSS (per Shapley method)
`GSS` = RMS_i(|φ(PC)|−|φ(LiNGAM)|)/output_range per feature, averaged over features; `SSS` = sign agreement PC vs LiNGAM; `SignDisagree` = 1 − SSS (the scatter's y-axis).

| Method | GSS | SSS | SignDisagree |
| --- | --- | --- | --- |
| Asymmetric | 0.0019 | 0.9407 | 0.0593 |
| Causal | 0.0129 | 0.6206 | 0.3794 |
| Flow | 0.0110 | 0.8673 | 0.1327 |


## Alignment to True — TGA vs Sign Alignment (per Shapley method × discovered graph)
`TGA` = RMS_i(|φ(disc)|−|φ(True)|)/output_range per feature, averaged over features; `SignAlign` = sign agreement vs True; `SignDisagree` = 1 − SignAlign (the scatter's y-axis).

| Method | Graph | Reference | TGA | SignAlign | SignDisagree |
| --- | --- | --- | --- | --- | --- |
| Asymmetric | PC | True | 0.0018 | 0.9412 | 0.0588 |
| Asymmetric | LiNGAM | True | 0.0014 | 0.9370 | 0.0630 |
| Causal | PC | True | 0.0096 | 0.6840 | 0.3160 |
| Causal | LiNGAM | True | 0.0137 | 0.6254 | 0.3746 |
| Flow | PC | True | 0.0133 | 0.5774 | 0.4226 |
| Flow | LiNGAM | True | 0.0146 | 0.5509 | 0.4491 |


## Alignment to Traditional — TGA vs Sign Alignment (per Shapley method × discovered graph)
`TGA` = RMS_i(|φ(disc)|−|φ(Traditional)|)/output_range per feature, averaged over features; `SignAlign` = sign agreement vs Traditional; `SignDisagree` = 1 − SignAlign (the scatter's y-axis).

| Method | Graph | Reference | TGA | SignAlign | SignDisagree |
| --- | --- | --- | --- | --- | --- |
| Asymmetric | PC | Traditional | 0.0012 | 0.9492 | 0.0508 |
| Asymmetric | LiNGAM | Traditional | 0.0016 | 0.9424 | 0.0576 |
| Causal | PC | Traditional | 0.0115 | 0.6470 | 0.3530 |
| Causal | LiNGAM | Traditional | 0.0106 | 0.6608 | 0.3392 |
| Flow | PC | Traditional | 0.0120 | 0.6342 | 0.3658 |
| Flow | LiNGAM | Traditional | 0.0120 | 0.6232 | 0.3768 |
