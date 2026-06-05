# Method-level group metrics — sachs

Scalar summaries behind the method-level scatter plots. Magnitude metrics (GSS, TGA) are the absolute average of the per-feature metric; sign metrics (SSS, Sign Alignment) are the mean over features.

## Graph-discovery instability — GSS vs SSS (per Shapley method)
`|GSS|` = magnitude difference PC vs LiNGAM; `SSS` = sign agreement PC vs LiNGAM; `SignDisagree` = 1 − SSS (the scatter's y-axis).

| Method | GSS | SSS | SignDisagree |
| --- | --- | --- | --- |
| Asymmetric | 4.8046 | 0.8140 | 0.1860 |
| Causal | 7.9310 | 0.6820 | 0.3180 |
| Flow | 9.2009 | 0.7812 | 0.2188 |


## Alignment to True — TGA vs Sign Alignment (per Shapley method × discovered graph)
`TGA` = magnitude difference vs True; `SignAlign` = sign agreement vs True; `SignDisagree` = 1 − SignAlign (the scatter's y-axis).

| Method | Graph | Reference | TGA | SignAlign | SignDisagree |
| --- | --- | --- | --- | --- | --- |
| Asymmetric | PC | True | 5.5502 | 0.7580 | 0.2420 |
| Asymmetric | LiNGAM | True | 1.8896 | 0.7660 | 0.2340 |
| Causal | PC | True | 6.9329 | 0.6370 | 0.3630 |
| Causal | LiNGAM | True | 6.4854 | 0.6770 | 0.3230 |
| Flow | PC | True | 9.2662 | 0.7164 | 0.2836 |
| Flow | LiNGAM | True | 4.8461 | 0.7619 | 0.2381 |


## Alignment to Traditional — TGA vs Sign Alignment (per Shapley method × discovered graph)
`TGA` = magnitude difference vs Traditional; `SignAlign` = sign agreement vs Traditional; `SignDisagree` = 1 − SignAlign (the scatter's y-axis).

| Method | Graph | Reference | TGA | SignAlign | SignDisagree |
| --- | --- | --- | --- | --- | --- |
| Asymmetric | PC | Traditional | 2.9141 | 0.8320 | 0.1680 |
| Asymmetric | LiNGAM | Traditional | 2.8097 | 0.8300 | 0.1700 |
| Causal | PC | Traditional | 5.1144 | 0.6870 | 0.3130 |
| Causal | LiNGAM | Traditional | 7.2298 | 0.6690 | 0.3310 |
| Flow | PC | Traditional | 8.9792 | 0.5980 | 0.4020 |
| Flow | LiNGAM | Traditional | 9.6142 | 0.5850 | 0.4150 |
