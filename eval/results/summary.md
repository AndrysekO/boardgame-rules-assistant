# Answer eval summary

Grade counts per graded run, read from the `grade` column of each CSV in `eval/results/answers/` (24 questions each).

| Run | Setup | Correct | Partial | Wrong | Other |
|---|---|---|---|---|---|
| baseline.csv | single search (model not recorded in CSV) | 14 | 3 | 7 | 0 |
| baseline-v2.csv | single search (model not recorded in CSV) | 11 | 4 | 9 | 0 |
| quote-first.csv | single search (model not recorded in CSV) | 12 | 4 | 7 | 1 ("correct, but the quote is wr") |
| agent-a.csv | Haiku 5.5, agent | 21 | 3 | 0 | 0 |
| agent-b.csv | Haiku 5.5, agent | 21 | 1 | 2 | 0 |
| agent-c.csv | Haiku 5.5, agent | 21 | 2 | 1 | 0 |

Not graded yet: haiku-quote-first.csv, haiku-quote-first-v2.csv.
