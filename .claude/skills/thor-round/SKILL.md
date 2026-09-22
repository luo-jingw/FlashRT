---
name: thor-round
description: Run one Thor validation round from the git-kept checklist: write the rows, hand them to the Thor agent, record its report, prune the checklist.
---

# Trigger

Use when a change or question needs measurements on the Jetson Thor (which has
no push credentials), and when a Thor report arrives.

# Inputs

Read:
- `THOR_CHECKLIST.md` (its "Usage" section is the contract);
- `THOR_STATUS_SUMMARY.md` (the round sections);
- `plan.md`, `opportunities.md`, `issues.md`;
- the report the Thor agent brought back.

# Procedure

## Write the rows

Each row is one question with: the command, the numbers to bring back, the
criterion (how to read them, not a guess at the answer), and the file the
conclusion goes to. Put a row whose failure makes the following rows
meaningless last, and do not make an unexplained result a stop condition for
the rest of the session.

## Hand over

The Thor pulls and runs; it never commits or pushes. Logs stay in `$OUT` (one
directory per round). Every number is reported with the commit, the clock
state, `emc_locked` and whether the GPU was idle.

## Read the report

- Compare only rows of one session, one harness and one prompt length. A gate
  P50 and an end-to-end P50 of the same configuration are different numbers
  (ISSUE-082).
- Bracket drift: repeat a reference row at the end; a row that moved more than
  the effect being measured is not a denominator.
- Check the printed `effective_config`: a row that fell back is void.
- Keep observation, evidence and hypothesis apart; label inference as inference.

## Record

- Measured numbers: the round section of `THOR_STATUS_SUMMARY.md`, and the
  `opportunities.md` entry they belong to.
- Defects and unexplained results: `issues.md` (Observation, Evidence,
  Hypotheses, Next Experiment).
- Decisions and phase status: `plan.md`.
- Delete every finished row from `THOR_CHECKLIST.md`; keep only what is undone.
- Commit by topic, push, no AI attribution.

# Completion

The checklist holds only undone rows; every recorded number names its round;
every hypothesis is labelled as one.
