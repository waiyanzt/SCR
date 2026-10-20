# results_compiled/FB15k237/ultra_4g

Per variant, on every query answerable on it:

| variant | queries | MRR | Hits@1 | Hits@10 |
| --- | --- | --- | --- | --- |
| inverse__compiled | 20466 | 0.3684 | 0.2723 | 0.5593 |
| original__compiled | 20466 | 0.3684 | 0.2723 | 0.5593 |
| reified_cvt_light__compiled | 20466 | 0.3684 | 0.2723 | 0.5593 |
| reified_cvt_strong__compiled | 20466 | 0.3684 | 0.2723 | 0.5593 |
| shortcut2_c200k__compiled | 20466 | 0.3684 | 0.2723 | 0.5593 |
| shortcut2_c50k__compiled | 20466 | 0.3684 | 0.2723 | 0.5593 |
| shortcut2_c5k__compiled | 20466 | 0.3684 | 0.2723 | 0.5593 |

Per pair, on the queries answerable on both (x 2 directions):

| pair | queries | Kendall tau | delta MRR | Hit@1 differs | Hit@10 differs | identical |
| --- | --- | --- | --- | --- | --- | --- |
| original__compiled vs inverse__compiled | 20466 | 1.0000 | +0.0000 | 0.000 | 0.000 | True |
| original__compiled vs reified_cvt_light__compiled | 20466 | 1.0000 | +0.0000 | 0.000 | 0.000 | True |
| original__compiled vs reified_cvt_strong__compiled | 20466 | 1.0000 | +0.0000 | 0.000 | 0.000 | True |
| original__compiled vs shortcut2_c200k__compiled | 20466 | 1.0000 | +0.0000 | 0.000 | 0.000 | True |
| original__compiled vs shortcut2_c50k__compiled | 20466 | 1.0000 | +0.0000 | 0.000 | 0.000 | True |
| original__compiled vs shortcut2_c5k__compiled | 20466 | 1.0000 | +0.0000 | 0.000 | 0.000 | True |
