# results_compiled/WN18RR/ultra_4g

Per variant, on every query answerable on it:

| variant | queries | MRR | Hits@1 | Hits@10 |
| --- | --- | --- | --- | --- |
| all_inverse_edges__compiled | 3134 | 0.4854 | 0.4146 | 0.6190 |
| no_changes__compiled | 3134 | 0.4854 | 0.4146 | 0.6190 |
| transitive_edges__compiled | 3134 | 0.4854 | 0.4146 | 0.6190 |
| universal_edges__compiled | 3134 | 0.4854 | 0.4146 | 0.6190 |

Per pair, on the queries answerable on both (x 2 directions):

| pair | queries | Kendall tau | delta MRR | Hit@1 differs | Hit@10 differs | identical |
| --- | --- | --- | --- | --- | --- | --- |
| no_changes__compiled vs all_inverse_edges__compiled | 3134 | 1.0000 | +0.0000 | 0.000 | 0.000 | True |
| no_changes__compiled vs transitive_edges__compiled | 3134 | 1.0000 | +0.0000 | 0.000 | 0.000 | True |
| no_changes__compiled vs universal_edges__compiled | 3134 | 1.0000 | +0.0000 | 0.000 | 0.000 | True |
