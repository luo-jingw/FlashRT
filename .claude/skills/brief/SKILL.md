---
name: brief
description: Write the six-field brief (why, readings, prior work, setup, cost and stop, result) before any experiment, survey, Thor round or task handed to a collaborator, and close it against its pre-stated readings.
---

# Trigger

Use before starting, or handing to someone else:
- an experiment or measurement (local, RTX 5090, Jetson Thor);
- a survey or literature check;
- a Thor round (`thor-round`) or a task for a collaborator;
- a request to read or summarise a collaborator's results (check them against
  the brief they were run under).

Use again when the result arrives, to fill field 6.

Not needed for a code change with a known correct outcome (`work`).

# Inputs

Read:
- the research line's main question (the topic `README.md` and the current
  task sheet, e.g. `research/topics/long-horizon-planner-executor/tasks_robocerebra.md`);
- `issues.md` and `opportunities.md` entries on the same subject;
- the topic's `papers.md`, `survey.md` and earlier experiment READMEs, for
  field 3.

# Procedure

Fill fields 1–5 before running anything. Write the brief at the top of the
experiment's `README.md`, the task entry, or the report, in that document's
language. Each field is short: a few lines or one table.

## 1. Why, and the link to the main question

- The main-line question this answers, quoted from the topic README or task
  sheet.
- What is missing without it: which conclusion has no evidence, or which
  decision cannot be made.
- If the link cannot be stated, do not run it. Record the idea in
  `opportunities.md` instead.

## 2. Measured quantities and pre-stated readings

- Every quantity to measure, with unit, sample count, machine and commit.
- A readings table, written before the run:

  | Outcome | What it shows | Next step |
  |---|---|---|

  At least one row states the outcome that would show the hypothesis is
  wrong.
- For a survey, the quantities are the questions to answer (for example
  "which systems inject planner latency and report SR"), and the readings
  table maps each possible answer to the decision it changes.
- Readings are stated as observations to compare, not as pass/fail
  thresholds on performance (AGENTS.md, Validation), unless a threshold is
  explicitly part of the task.

## 3. Prior work

- Existing experiments of the same kind: setting, numbers, source (paper and
  table or figure number; repository path for our own earlier runs).
- How ours differs: hardware, model, protocol, benchmark.
- Whether their numbers serve as a reference point, an upper bound, or are
  not comparable.
- If none exists, state the search that found none.

## 4. Setup

- Variables under test and their values.
- Fixed quantities with their values and the source of each value. A value
  inherited from another project or harness (a check interval, a token cap, a
  cache size, a run mode) is named as inherited, with its origin, and checked
  against the current main line.
- Definitions of the key quantities (for example where L starts and ends).
- The control or baseline.

## 5. Cost and stop conditions

- Expected machine time, and whether it uses a shared machine (Jetson Thor
  through the GPU lock).
- The conditions under which work stops and is reported instead of continued
  (a gate not met, a definition found ambiguous, a result outside every row
  of the readings table).

## 6. Result (after the run)

- Measured values.
- For each measured outcome, the row of the readings table it matches; an
  outcome matching no row is reported as such, not fitted to a row after the
  fact.
- Conclusion and confidence, kept separate from the numbers.
- Where each finding goes: verified results to `docs/` or the experiment
  `README.md`; unexplained results to `issues.md`; unexplored cells to
  `opportunities.md`.

# Template

For documents under `research/` (written in Chinese):

```markdown
## 实验说明

1. **为什么做，和主线的关系**：回答主线上的哪个问题（引用 topic README 或任务单原文）；不做会缺哪条证据或哪个决定。
2. **指标与判读**：要测的量（单位、样本数、机器、commit）。

   | 结果 | 说明什么 | 下一步 |
   |---|---|---|
   | … | … | … |
   | （推翻假设的结果） | … | … |

3. **已有工作**：同类实验的设置、数值与出处（论文表号或图号）；与我们的差别；能否作为对照或上界。
4. **设置**：变量与取值；固定量、取值与来源（沿用自其他项目的取值注明来源）；关键量的定义；对照。
5. **代价与停止条件**：预计机时、是否占用共享 Thor；遇到什么情况停下报告。
6. **结果**（做完后填）：实测值；每个结果对应判读表的哪一行；结论与置信度；去向（docs / README / issues / opportunities）。
```

# Relation to other skills

- `experiment` runs the grid after fields 1–5 exist; field 4 is its fixed
  variables and levels.
- `thor-round` rows carry fields 1, 2 and 4 in short form; the round report
  fills field 6.
- A task for a collaborator states fields 1–5; the collaborator's report
  answers field 6 against field 2. When reading such a report, check fields
  1 and 4 first: a result produced under an inherited or off-line setting is
  scoped to that setting.

# Stop Conditions

Stop and report when:
- field 1 cannot be linked to the main question;
- a fixed quantity in field 4 has no known source;
- a field 2 quantity has no defined measurement method.

# Completion

A brief is complete when fields 1–5 were written before the run and field 6
compares every measured outcome with the pre-stated readings.
