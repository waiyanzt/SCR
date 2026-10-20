# results_compiled/Hetionet/ultra_4g

Per variant, on every query answerable on it:

| variant | queries | MRR | Hits@1 | Hits@10 |
| --- | --- | --- | --- | --- |
| inverse__compiled | 5000 | 0.2741 | 0.2102 | 0.4026 |
| original__compiled | 5000 | 0.2741 | 0.2102 | 0.4026 |
| reified_regulation__compiled | 5000 | 0.2741 | 0.2102 | 0.4026 |
| shortcut2_c50k__compiled | 5000 | 0.2741 | 0.2102 | 0.4026 |

Per pair, on the queries answerable on both (x 2 directions):

| pair | queries | Kendall tau | delta MRR | Hit@1 differs | Hit@10 differs | identical |
| --- | --- | --- | --- | --- | --- | --- |
| original__compiled vs inverse__compiled | 5000 | 1.0000 | +0.0000 | 0.000 | 0.000 | True |
| original__compiled vs reified_regulation__compiled | 5000 | 1.0000 | +0.0000 | 0.000 | 0.000 | True |
| original__compiled vs shortcut2_c50k__compiled | 5000 | 1.0000 | +0.0000 | 0.000 | 0.000 | True |
