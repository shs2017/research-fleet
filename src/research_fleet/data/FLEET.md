<!--
How this project's files and directories work. Fleet prepends this file to every agent
job's system prompt, so it never needs repeating in a task prompt.

Keep it about the layout: where things are, what is writable, what to do with what an
earlier stage or run left behind. What a job should *conclude*, and to what standard,
belongs in that job's task prompt. Edit freely; it is read fresh at the start of a run.
-->

## How a run is organised

Work here is a **workflow**: a sequence of **stages**, each a separate agent in its own
container. You are one stage. The directories you actually have are listed above under
"Files and directories"; this is what they mean.

- **`/results` is your deliverable.** It is private to your job, and it is the only
  place your work survives. Write finished output there, not just into `/workspace`.
- **`/inputs/<stage>`** is an earlier stage of *this* run, read-only. A path an earlier
  stage calls `/results/x` is yours at `/inputs/<that stage>/x` — the two are different
  directories, so check there before reporting a promised file missing.
- **`/previous/<stage>`** is the same layout for an earlier *run* of this workflow: a
  whole prior attempt, present only when this run was told to build on one.

By convention a stage directory holds its written deliverable at the top level, any
scripts under `code/`, and `output.md`, the final message that stage reported. Where
both exist, read the deliverable rather than `output.md` — the message is a summary of
it, and may be truncated.

## Working with what earlier stages left

- Read what is in `/inputs/` and `/previous/` before starting. It exists so you do not
  repeat it.
- Those directories are read-only. To run or modify earlier code, copy it into
  `/workspace` first, and say that you did.
- Cite the file you took something from, so a later stage can trace it.
