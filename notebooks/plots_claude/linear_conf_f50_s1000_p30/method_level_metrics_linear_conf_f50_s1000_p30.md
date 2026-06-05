# Method-level group metrics — linear_conf_f50_s1000_p30

Scalar summaries behind the method-level scatter plots. Magnitude metrics (GSS, TGA) are the absolute average of the per-feature metric; sign metrics (SSS, Sign Alignment) are the mean over features.

## Graph-discovery instability — GSS vs SSS (per Shapley method)
`|GSS|` = magnitude difference PC vs LiNGAM; `SSS` = sign agreement PC vs LiNGAM; `SignDisagree` = 1 − SSS (the scatter's y-axis).

| Method | GSS | SSS | SignDisagree |
| --- | --- | --- | --- |
| Asymmetric | 0.0385 | 0.9407 | 0.0593 |
| Causal | 0.2634 | 0.6206 | 0.3794 |
| Flow | 0.2159 | 0.8673 | 0.1327 |


## Alignment to True — TGA vs Sign Alignment (per Shapley method × discovered graph)
`TGA` = magnitude difference vs True; `SignAlign` = sign agreement vs True; `SignDisagree` = 1 − SignAlign (the scatter's y-axis).

| Method | Graph | Reference | TGA | SignAlign | SignDisagree |
| --- | --- | --- | --- | --- | --- |
| Asymmetric | PC | True | 0.0369 | 0.9412 | 0.0588 |
| Asymmetric | LiNGAM | True | 0.0286 | 0.9370 | 0.0630 |
| Causal | PC | True | 0.1960 | 0.6840 | 0.3160 |
| Causal | LiNGAM | True | 0.2784 | 0.6254 | 0.3746 |
| Flow | PC | True | 0.2625 | 0.5774 | 0.4226 |
| Flow | LiNGAM | True | 0.2866 | 0.5509 | 0.4491 |


## Alignment to Traditional — TGA vs Sign Alignment (per Shapley method × discovered graph)
`TGA` = magnitude difference vs Traditional; `SignAlign` = sign agreement vs Traditional; `SignDisagree` = 1 − SignAlign (the scatter's y-axis).

| Method | Graph | Reference | TGA | SignAlign | SignDisagree |
| --- | --- | --- | --- | --- | --- |
| Asymmetric | PC | Traditional | 0.0242 | 0.9492 | 0.0508 |
| Asymmetric | LiNGAM | Traditional | 0.0309 | 0.9424 | 0.0576 |
| Causal | PC | Traditional | 0.2381 | 0.6470 | 0.3530 |
| Causal | LiNGAM | Traditional | 0.2190 | 0.6608 | 0.3392 |
| Flow | PC | Traditional | 0.2431 | 0.6342 | 0.3658 |
| Flow | LiNGAM | Traditional | 0.2448 | 0.6232 | 0.3768 |
