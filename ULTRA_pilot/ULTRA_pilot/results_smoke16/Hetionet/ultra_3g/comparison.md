# results_smoke16/Hetionet/ultra_3g

Common queries: 16 (x 2 directions)

| variant | MRR | Hits@1 | Hits@10 |
| --- | --- | --- | --- |
| inverse | 0.3365 | 0.2500 | 0.5625 |
| original | 0.3289 | 0.2500 | 0.5625 |
| original__repeat | 0.3289 | 0.2500 | 0.5625 |
| reified_regulation | 0.3726 | 0.3125 | 0.5312 |
| shortcut2_c50k | 0.3161 | 0.2500 | 0.5312 |

| pair | Kendall tau | delta MRR | Hit@1 differs | Hit@10 differs | identical |
| --- | --- | --- | --- | --- | --- |
| original vs inverse | 0.8437 | +0.0076 | 0.062 | 0.125 | False |
| original vs original__repeat | 1.0000 | +0.0000 | 0.000 | 0.000 | True |
| original vs reified_regulation | 0.9149 | +0.0438 | 0.062 | 0.031 | False |
| original vs shortcut2_c50k | 0.7316 | -0.0128 | 0.062 | 0.031 | False |
