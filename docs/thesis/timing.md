### timing — requirements of section 1.3.3

Machine: x86_64, 4 threads, CPU only, torch 2.14.0+cpu. Queue: 3 runs; justification: the first 10 cards of the queue.

| config | changes in window | queue, median s | queue, max s | justification, median s | justification, max s | top 50 same set | top 50 same order |
|---|---|---|---|---|---|---|---|
| synthetic.yaml | 637 | 0.79 | 0.93 | 1.41 | 1.47 | yes | yes |
| full.yaml | 2759 | 4.58 | 4.72 | 5.30 | 6.37 | yes | yes |

The queue time includes building the window from the source: for the
synthetic configurations that is generating the dataset, for a live engine it
is reading the graph over Bolt. The model is trained beforehand and loaded once.
