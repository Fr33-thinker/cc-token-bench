# cc-token-bench: design spec

Date: 2026-09-02
Status: approved design, awaiting spec review
Owner: Benny Pan

## 1. Purpose

A reusable, publishable benchmark harness that measures how many tokens, how
many dollars, how many turns, and how much wall-clock time Claude Code spends to
complete a fixed coding task, across a grid of models and effort levels, with
pass or fail decided by hidden tests.

The first study compares Claude Fable 5 and Claude Fable 5.1 at effort low,
medium, high, and xhigh. Later studies add new Anthropic models by editing one
matrix file. Results from different studies are comparable when the task
fingerprints match.

## 2. Non-goals

- Not a general LLM benchmark. It measures the Claude Code CLI as a user
  experiences it, including its system prompt and default tools.
- Not a quality-of-code judge. Pass or fail comes from tests only.
- No API-key billing. Runs use the subscription. Cost is Claude Code's own
  list-price estimate (see facts below).
- No cross-provider support.

## 3. Facts verified on 2026-09-02 (Claude Code 2.1.258, macOS)

These were checked by running probes, not recalled from memory.

| Fact | Evidence |
|---|---|
| `claude -p ... --output-format json` returns `total_cost_usd`, per-model `costBasis: "list"`, and usage split into input, cache_creation, cache_read, output, and thinking tokens | Haiku probe output |
| Cost is priced from the public list price even on a subscription | `costBasis: "list"` in `modelUsage` |
| `--output-format stream-json --verbose` in `-p` mode emits one JSON event per line: `system/init`, `assistant`, `user` (tool results), `system/thinking_tokens`, `rate_limit_event`, and a final `result` event whose fields match the plain json output | Haiku stream probe |
| `rate_limit_event` carries `status` (`allowed` or not), `rateLimitType`, `resetsAt`, and `unifiedWindows.five_hour.utilization` | Haiku stream probe |
| `--effort` accepts `low`, `medium`, `high`, `xhigh`, `max` | `claude --help` |
| `--permission-mode bypassPermissions` works in `-p` mode and lets Claude write files unattended | Haiku stream probe created a file |
| Benny's global config adds about 44,000 tokens of context to a trivial run (132 slash commands, 16 agents, hooks, CLAUDE.md, memory) | init event + usage in probe |
| Setting `CLAUDE_CONFIG_DIR` to a fresh folder gives a clean session but starts logged out. Login is stored per config folder (Keychain entry `Claude Code-credentials-<hash>`) | Empty-config probe returned "Not logged in" |
| Claude Code default effort is xhigh | Benny's settings show `effortLevel: xhigh` per model |

## 4. Repository layout

```
cc-token-bench/
  README.md                 how to reproduce a study in five commands
  pyproject.toml            package "bench", Python 3.11+, no runtime deps
                            (pytest for tests; matplotlib optional for charts)
  bench/                    the harness package
    cli.py                  entry point: bench <subcommand>
    matrix.py               load + validate a study matrix (TOML)
    tasks.py                task discovery, fingerprint, fresh-copy
    preflight.py            environment checks + baseline overhead runs
    runner.py               one run = fresh copy + claude -p + capture
    scorer.py               hidden tests + cheat check
    report.py               aggregate to CSV, markdown, chart
    claude_cli.py           builds the claude command line, parses events
    schema.py               dataclasses for meta.json and cell summaries
  tasks/
    ledger-bugfix/          task A (bug fix)
      task.toml             name, version, difficulty, expected minutes
      PROMPT.md             the exact prompt Claude receives, verbatim
      template/             the code Claude sees, incl. visible tests
      hidden_tests/         decides pass/fail; never copied into template
    notes-feature/          task B (feature)
      (same layout)
  studies/
    smoke-opus5.toml        1 run per task on Opus 5, validates the pipeline
    2026-09-fable5-vs-fable51.toml
  results/
    <study-id>/
      environment.json      cc version, os, python, date, task fingerprints
      baseline/<model>.json fixed overhead per model
      runs/<run-id>/
        meta.json           everything the report needs (schema in section 9)
        stream.jsonl        full event stream
        result.json         the final result event
        diff.patch          what Claude changed
        stderr.log
        junit.xml           hidden test results (after scoring)
      summary.csv
      summary.md
      summary.png           (only if matplotlib is installed)
  tests/                    harness tests; use a fake claude binary
  docs/superpowers/specs/   this file
```

The benchmark config folder lives outside the repo at `~/.cc-bench` by default,
overridable with `BENCH_CONFIG_DIR`. It is never committed.

## 5. Components

### 5.1 Matrix file (`studies/*.toml`)

TOML is used because Python 3.11 reads it with the standard library, so the
harness has zero runtime dependencies.

```toml
study_id = "2026-09-fable5-vs-fable51"
claude_code_version = "2.1.258"      # preflight refuses on mismatch
models  = ["claude-fable-5", "claude-fable-5-1"]
efforts = ["low", "medium", "high", "xhigh"]
tasks   = ["ledger-bugfix", "notes-feature"]
repeats = 3
parallel = 2
run_timeout_minutes = 45
budget_usd = 500                     # list-price cap; runner stops when exceeded
shuffle_seed = 20260902
```

`matrix.py` validates every field, expands the grid into run ids, and shuffles
the order with the recorded seed. Run id format:
`{model}__{effort}__{task}__r{repeat}`.

### 5.2 Tasks

Each task is a self-contained Python codebase of roughly 300 to 500 lines using
only the standard library, so a fresh copy needs no install step. Sized for 30
to 60 minutes of junior-engineer work.

- **ledger-bugfix.** A small expense-ledger library (CSV in, reports out).
  Three planted bugs: an off-by-one in a date-range filter, wrong rounding in
  multi-currency totals, and a mutable default argument that leaks state
  between calls. The prompt describes the user-visible symptoms, not the bugs.
  Visible tests pass on the broken code. Hidden tests fail until all three are
  fixed.
- **notes-feature.** A small command-line notes tool. The prompt is a short
  feature request: add tags, filter by tag, and a `stats` command, with the
  exact output format specified. Hidden tests check behaviour against that
  format.

`task.toml` holds `name`, `version`, `difficulty`, `expected_minutes`, and
`test_paths` (the visible test files, used by the cheat check).

`tasks.py` computes a fingerprint: SHA-256 over the sorted relative paths and
contents of `template/`, `hidden_tests/`, and `PROMPT.md`. The fingerprint is
recorded in `environment.json` and in every `meta.json`. Two studies are
comparable for a task only when the fingerprints are equal.

### 5.3 Preflight (`bench preflight <matrix>`)

Refuses to start the study unless all of these hold, and prints every failure
in plain words:

1. `claude --version` equals `claude_code_version` in the matrix.
2. The benchmark config folder exists and a one-line Haiku probe returns
   `is_error: false` (proves login works).
3. The benchmark config folder contains no `CLAUDE.md`, no `plugins/`, no
   `hooks` or `mcpServers` keys in its `settings.json`. Warns if
   `settings.json` sets `model` or `effortLevel`, because the runner overrides
   both on the command line anyway.
4. Every task in the matrix exists and its fingerprint matches the value in
   `results/<study-id>/environment.json` if that file already exists (resume
   safety).
5. Python is 3.11 or newer and `pytest` is importable.

Then it runs one baseline call per model in the matrix: prompt "Reply with
exactly the word OK and nothing else." in an empty folder, and stores the
usage and cost as `results/<study-id>/baseline/<model>.json`. This is the fixed
overhead of starting Claude Code with that model. Baselines are skipped if the
file already exists.

Preflight writes `environment.json` with the Claude Code version, OS, Python
version, date, matrix contents, and task fingerprints.

### 5.4 Runner (`bench run <matrix> [--parallel N] [--only RUN_ID]`)

For each run id in shuffled order, up to `parallel` at a time:

1. Skip if `results/<study-id>/runs/<run-id>/meta.json` exists with
   `status = "completed"`. This is what makes the study resumable.
2. Create a fresh work folder under the system temp directory, copy
   `tasks/<task>/template/` into it, `git init`, commit everything as
   `baseline`. Hidden tests are never copied here.
3. Start Claude Code with the work folder as the current directory:

   ```
   CLAUDE_CONFIG_DIR=~/.cc-bench claude -p "<PROMPT.md contents>" \
     --model <model> --effort <effort> \
     --output-format stream-json --verbose \
     --permission-mode bypassPermissions \
     --strict-mcp-config --no-session-persistence
   ```

   Stdout goes to `stream.jsonl`, stderr to `stderr.log`. Wall-clock is
   measured around the process. The process is killed after
   `run_timeout_minutes`.
4. After exit, `git add -A && git diff --cached` becomes `diff.patch`. The last
   `result` event is saved as `result.json`. The work folder is kept until
   scoring finishes, then deleted.
5. Classify the run status:
   - `completed`: result event present, `is_error` false, `terminal_reason`
     `completed`.
   - `rate_limited`: any `rate_limit_event` with `status` other than
     `allowed`, or a result whose text mentions a rate or usage limit. The
     runner stops launching new runs, prints the `resetsAt` time in local
     time, and exits non-zero. The run folder is deleted so it reruns on the
     next `bench run`.
   - `timeout`: killed by the runner.
   - `api_error`: result event `is_error` true for any other reason.
   - `harness_error`: exception in the runner itself (traceback saved).
6. Write `meta.json` (section 9). Add the run's `total_cost_usd` to a running
   total; if it exceeds `budget_usd`, stop launching new runs and say so.
7. Print one line per run: run id, status, tokens, cost, turns, minutes, and
   the five-hour rate-limit utilization from the last `rate_limit_event`.

`--only RUN_ID` reruns one run after deleting its folder, for debugging.

### 5.5 Scorer (`bench score <matrix>`)

For every run with status `completed` and no `pass` field yet:

1. Copy `tasks/<task>/hidden_tests/` into `<workdir>/_hidden_tests/` (the work
   folder is still present, because scoring runs right after the run inside
   the runner; `bench score` as a separate command re-creates the work folder
   by applying `diff.patch` onto a fresh template copy).
2. Run `python -m pytest _hidden_tests -q --junitxml=junit.xml` with the work
   folder as the current directory and on `PYTHONPATH`, 120-second timeout.
3. Parse `junit.xml` with the standard library. Record `tests_total`,
   `tests_passed`, and `pass = (tests_passed == tests_total)`.
4. Cheat check: hash every file listed in `task.toml` `test_paths` in the work
   folder and compare with the template. Record `visible_tests_modified`.
   The run is still scored; the report shows the flag.
5. Record `diff_files_changed` and `diff_lines` from `diff.patch`.

Runs with any other status get `pass = null`.

### 5.6 Report (`bench report <study-id> [--studies A B ...]`)

A cell is one (model, effort, task). For each cell, over runs with status
`completed`:

| Column | Definition |
|---|---|
| n_completed, n_pass, pass_rate | counts and n_pass / n_completed |
| n_other | runs with status timeout, rate_limited, api_error, harness_error |
| tokens_total | input + cache_creation + cache_read + output, median and min to max |
| input, cache_creation, cache_read, output, thinking | each median |
| cost_usd | median and min to max |
| cost_per_pass | sum of cost_usd over completed runs / n_pass; "no passes" if n_pass is 0 |
| turns, minutes | median |
| baseline_cost_usd | the model's baseline overhead, for context |
| tests_modified | count of runs with visible_tests_modified true |

Outputs:

- `summary.csv` with one row per cell plus the per-run raw rows in
  `runs.csv`.
- `summary.md` with a table per task and a short methods block (Claude Code
  version, date, n, task fingerprints, seed) ready to paste into a post.
- `summary.png` if matplotlib is installed: cost per pass by effort, one line
  per model, one panel per task. Skipped with a notice otherwise.

`--studies` merges several studies into one report and refuses, naming the
task, if any task fingerprint differs between them.

### 5.7 CLI

`bench` has subcommands `login`, `preflight`, `run`, `score`, `report`, and
`all`. `login` only prints the two commands the user runs by hand once
(`CLAUDE_CONFIG_DIR=~/.cc-bench claude`, then `/login`, then `/exit`) because
login is interactive. `all` runs preflight, run, score, report in sequence.

## 6. Isolation

The benchmark config folder contains only what Claude Code writes on first
login. No `CLAUDE.md`, memory, plugins, hooks, output style, or MCP servers.
Preflight enforces this. Everything else stays at Claude Code defaults,
because "Claude Code out of the box" is the thing being measured. The runner
passes `--strict-mcp-config` so a stray project-level MCP config cannot leak
in, and the task templates contain no `.claude/` folder or `CLAUDE.md`.

## 7. Error handling

- Every failure mode in the runner is a recorded status, never a crash of the
  whole study. Only harness bugs raise, and their traceback lands in the run
  folder.
- Rate limits stop the study cleanly and print when to resume.
- The budget cap is checked before each new run starts, so it can overshoot by
  at most `parallel` runs.
- `bench score` and `bench report` are pure functions of the files on disk.
  They can be rerun any time.

## 8. Testing the harness

A fake `claude` executable in `tests/fake_claude/` replaces the real one via
`BENCH_CLAUDE_BIN`. It reads a scenario name from `FAKE_CLAUDE_SCENARIO` and
emits a canned event stream: `ok` (writes a correct fix), `wrong` (writes a
broken fix), `edits_tests` (modifies a visible test), `rate_limited`,
`api_error`, and `hang` (sleeps past the timeout). Tests cover:

- matrix validation, grid expansion, seeded shuffle order stability
- task fingerprint stability and change detection
- runner: fresh copy has no hidden tests, meta.json written, resume skips
  completed runs, rate limit stops the batch, timeout kills, budget cap stops
- scorer: pass, fail, cheat flag, re-creation from diff.patch
- report: cell math on a fixture study, cost_per_pass with zero passes, refusal
  on fingerprint mismatch
- preflight: version mismatch refuses, dirty config folder refuses

No test calls the real Claude Code. The real pipeline is validated once by the
Opus 5 smoke study before the Fable study.

## 9. Per-run record (`meta.json`)

```
study_id, run_id, model, effort, task, task_version, task_fingerprint, repeat,
order_index, shuffle_seed, claude_code_version,
started_at, ended_at, duration_ms, duration_api_ms,
status, terminal_reason, error_text,
num_turns, input_tokens, cache_creation_input_tokens, cache_read_input_tokens,
output_tokens, thinking_tokens, total_cost_usd, cost_basis,
rate_limit_utilization_5h,
pass, tests_total, tests_passed, visible_tests_modified,
diff_files_changed, diff_lines
```

## 10. Study protocol (the runbook that ships in README)

1. One time: `bench login` and follow the two printed commands.
2. `bench preflight studies/smoke-opus5.toml` then `bench all studies/smoke-opus5.toml`. Read `summary.md`. Both tasks must pass; if Opus 5 at default effort cannot pass a task, the task is broken or too hard and gets revised before the Fable study.
3. `bench all studies/2026-09-fable5-vs-fable51.toml --parallel 2`. Expect 4 to 8 hours. If it stops on a rate limit, rerun the same command after the printed reset time; finished runs are skipped.
4. Commit `results/<study-id>/` in full. Write the post from `summary.md`.
5. For a new model later: copy the matrix file, change `study_id` and `models`, keep `claude_code_version` current, rerun steps 2 and 3, then `bench report --studies old new`.

## 11. Caveats the write-up must state

- Public tasks may enter future training data. Task version and fingerprint are published so readers can judge cross-study comparisons.
- Three repeats gives a range, not a confidence interval. Label results pilot-scale.
- Cost is Claude Code's list-price estimate, not a bill.
- Effort `max` is excluded from the first study on cost grounds; the harness supports it.
- Results depend on the pinned Claude Code version. A new version is a new study.

## 12. Out of scope for now

Multiple Claude Code versions in one study, non-Python tasks, LLM-judged code
quality, and a hosted results dashboard.
