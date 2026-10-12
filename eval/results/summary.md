# Answer eval summary

Grade counts per graded run, read from the `grade` column of each CSV in `eval/results/answers/` (24 questions each).

| Run | Model | Correct | Partial | Wrong | Bad question |
|---|---|---|---|---|---|
| baseline.csv | llama3.1:8b | 14 | 3 | 7 | 0 |
| baseline-v2.csv | llama3.1:8b | 11 | 4 | 9 | 0 |
| quote-first.csv | llama3.1:8b | 13 | 4 | 7 | 0 |
| haiku-quote-first.csv | claude-haiku-5-5 | 16 | 2 | 5 | 1 |
| haiku-quote-first-v2.csv | claude-haiku-5-5 | 15 | 3 | 6 | 0 |
| agent-a.csv | claude-haiku-5-5 | 21 | 3 | 0 | 0 |
| agent-b.csv | claude-haiku-5-5 | 21 | 1 | 2 | 0 |
| agent-c.csv | claude-haiku-5-5 | 21 | 2 | 1 | 0 |
