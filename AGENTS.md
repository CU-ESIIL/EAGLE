# AGENTS.md

Keep the working documents up to date with progress, planned next steps, completed items, and open questions: `training_working_doc.md` (pre-training) and `benchmarking_working_doc.md` (baseline benchmark). They replace the claude.ai working-notes artifact (retired 2026-11-12).

use pixi for environment management

keep working .md docs with status/plan/hand-off notes until a task is complete; don't leave stale notes hanging around. Since project happens across 5 compute platforms, some with finite-duration jobs, expect that some tasks will be interrupted and need to be resumed by another agent. Operations involving storing/using large datasets will only be run on clusters (PSC bridges2 and NCSA delta).

Prefer improving shared systems over one-off fixes.

Keep documentation synchronized with repository structure.

Policy for using compute resources on clusters (bridges2, delta): ask me if the task will consume more than 300 CPU-hours or 30 GPU hours, otherwise use your discretion for running jobs on GPU and CPU nodes. If my cpu or gpu credits reach 80% usage, check with me before running any slurm jobs for that partition. 