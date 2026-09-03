# cc-token-bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a zero-dependency Python harness that runs Claude Code on fixed coding tasks across a model × effort grid, scores each run with hidden tests, and reports tokens, dollars, turns, minutes, pass rate and cost per passed task.

**Architecture:** A `bench` package with one module per stage (matrix → tasks → runner → scorer → report, plus preflight and a CLI). Every run gets a fresh temporary work folder; Claude Code is driven through its real command line in non-interactive stream mode with an isolated config folder. All harness tests use a fake `claude` executable so the test suite never spends money.

**Tech Stack:** Python 3.11+ (stdlib only at runtime: `tomllib`, `subprocess`, `json`, `csv`, `statistics`, `xml.etree`), `pytest` for tests, `matplotlib` optional for the chart, `git` on PATH.

**Spec:** `docs/superpowers/specs/2026-09-02-cc-token-bench-design.md`

## Global Constraints

- Python `>=3.11`; runtime dependencies: none. `pytest>=8` is a dev dependency; `matplotlib` is optional.
- The Claude Code command line is fixed: `claude -p <prompt> --model <m> --effort <e> --output-format stream-json --verbose --permission-mode bypassPermissions --strict-mcp-config --no-session-persistence`.
- Effort values: `low`, `medium`, `high`, `xhigh`, `max`.
- Benchmark config folder: `$BENCH_CONFIG_DIR`, default `~/.cc-bench`. Never inside the repo. Runner exports it as `CLAUDE_CONFIG_DIR`.
- The `claude` executable path comes from `$BENCH_CLAUDE_BIN`, default `claude`. Tests point it at `tests/fake_claude/claude`.
- Hidden tests are never copied into a work folder before the run. They are copied to `<workdir>/_hidden_tests/` only at scoring time, after the diff is captured.
- Run id format: `{model}__{effort}__{task}__r{repeat}`. Results live at `results/<study_id>/runs/<run_id>/`.
- Run statuses: `completed`, `rate_limited`, `timeout`, `api_error`, `harness_error`.
- JSON key for pass/fail is `passed` (the spec says `pass`; that is a Python keyword, so the field is named `passed` everywhere).
- Task templates must not contain `hidden_tests/`, `CLAUDE.md`, or `.claude/`.
- Commit after every task. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Work from the repo root `~/Documents/GitHub/cc-token-bench` with the venv active: `source .venv/bin/activate`.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `bench/__init__.py`, `tests/test_scaffold.py`

**Interfaces:**
- Produces: importable package `bench`, `bench.__version__`, console script `bench` (wired to `bench.cli:main`, which Task 10 creates).

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "cc-token-bench"
version = "0.1.0"
description = "Reproducible token-efficiency benchmark for Claude Code across models and effort levels"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8"]
chart = ["matplotlib>=3.8"]

[project.scripts]
bench = "bench.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["bench*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write .gitignore and package init**

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
build/
dist/
.DS_Store
```

`bench/__init__.py`:
```python
"""cc-token-bench: measure Claude Code token efficiency across models and effort levels."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Write the failing scaffold test**

`tests/test_scaffold.py`:
```python
import bench


def test_package_has_version():
    assert bench.__version__ == "0.1.0"
```

- [ ] **Step 4: Create the venv, install, run the test**

Run:
```bash
cd ~/Documents/GitHub/cc-token-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
```
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore bench/__init__.py tests/test_scaffold.py
git commit -m "chore: project scaffold

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Per-run record (`bench/schema.py`)

**Files:**
- Create: `bench/schema.py`, `tests/test_schema.py`

**Interfaces:**
- Produces: `RunMeta` dataclass (fields listed below), `RunMeta.tokens_total` property, `RunMeta.to_json() -> str`, `RunMeta.from_json(text) -> RunMeta`, `RunMeta.save(path)`, `RunMeta.load(path) -> RunMeta`, constant `STATUSES`.

- [ ] **Step 1: Write the failing tests**

`tests/test_schema.py`:
```python
from pathlib import Path

from bench.schema import STATUSES, RunMeta


def make_meta(**overrides) -> RunMeta:
    base = dict(
        study_id="s", run_id="m__low__t__r1", model="m", effort="low", task="t",
        task_version="1", task_fingerprint="abc", repeat=1, order_index=0,
        shuffle_seed=7, claude_code_version="2.1.258",
    )
    base.update(overrides)
    return RunMeta(**base)


def test_defaults_are_unknown():
    m = make_meta()
    assert m.status == "harness_error"
    assert m.passed is None
    assert m.tokens_total is None


def test_tokens_total_sums_four_buckets():
    m = make_meta(input_tokens=1, cache_creation_input_tokens=2,
                  cache_read_input_tokens=3, output_tokens=4)
    assert m.tokens_total == 10


def test_json_round_trip(tmp_path: Path):
    m = make_meta(status="completed", passed=True, total_cost_usd=1.25)
    p = tmp_path / "meta.json"
    m.save(p)
    assert RunMeta.load(p) == m
    assert '"passed": true' in p.read_text()


def test_from_json_ignores_unknown_keys():
    m = make_meta()
    text = m.to_json().replace('"study_id"', '"future_field": 1, "study_id"')
    assert RunMeta.from_json(text) == m


def test_statuses():
    assert STATUSES == ("completed", "rate_limited", "timeout", "api_error", "harness_error")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_schema.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.schema'`.

- [ ] **Step 3: Write the implementation**

`bench/schema.py`:
```python
"""One RunMeta = one meta.json = everything the report needs about one run."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

STATUSES = ("completed", "rate_limited", "timeout", "api_error", "harness_error")


@dataclass
class RunMeta:
    # identity
    study_id: str
    run_id: str
    model: str
    effort: str
    task: str
    task_version: str
    task_fingerprint: str
    repeat: int
    order_index: int
    shuffle_seed: int
    claude_code_version: str
    # timing
    started_at: str = ""
    ended_at: str = ""
    duration_ms: int = 0
    duration_api_ms: int | None = None
    # outcome
    status: str = "harness_error"
    terminal_reason: str | None = None
    error_text: str | None = None
    # usage (from the final result event)
    num_turns: int | None = None
    input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    total_cost_usd: float | None = None
    cost_basis: str | None = None
    rate_limit_utilization_5h: float | None = None
    # scoring
    passed: bool | None = None
    tests_total: int | None = None
    tests_passed: int | None = None
    visible_tests_modified: bool | None = None
    diff_files_changed: int | None = None
    diff_lines: int | None = None

    @property
    def tokens_total(self) -> int | None:
        parts = (self.input_tokens, self.cache_creation_input_tokens,
                 self.cache_read_input_tokens, self.output_tokens)
        if any(p is None for p in parts):
            return None
        return sum(parts)  # type: ignore[arg-type]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RunMeta":
        data = json.loads(text)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json() + "\n")

    @classmethod
    def load(cls, path: Path) -> "RunMeta":
        return cls.from_json(Path(path).read_text())
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_schema.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add bench/schema.py tests/test_schema.py
git commit -m "feat: RunMeta per-run record

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Study matrix (`bench/matrix.py`)

**Files:**
- Create: `bench/matrix.py`, `tests/test_matrix.py`

**Interfaces:**
- Produces: `Matrix` frozen dataclass (`study_id, claude_code_version, models, efforts, tasks, repeats, parallel, run_timeout_minutes, budget_usd, shuffle_seed`), `RunSpec` frozen dataclass (`run_id, model, effort, task, repeat, order_index`), `load_matrix(path) -> Matrix`, `expand_runs(matrix) -> list[RunSpec]`, `make_run_id(model, effort, task, repeat) -> str`, `MatrixError`, `VALID_EFFORTS`.

- [ ] **Step 1: Write the failing tests**

`tests/test_matrix.py`:
```python
import pytest

from bench.matrix import MatrixError, expand_runs, load_matrix, make_run_id

GOOD = '''
study_id = "demo"
claude_code_version = "2.1.258"
models = ["claude-fable-5", "claude-fable-5-1"]
efforts = ["low", "xhigh"]
tasks = ["ledger-bugfix", "notes-feature"]
repeats = 3
parallel = 2
run_timeout_minutes = 45
budget_usd = 500
shuffle_seed = 20260902
'''


def write(tmp_path, text):
    p = tmp_path / "m.toml"
    p.write_text(text)
    return p


def test_load_good(tmp_path):
    m = load_matrix(write(tmp_path, GOOD))
    assert m.study_id == "demo"
    assert m.models == ("claude-fable-5", "claude-fable-5-1")
    assert m.repeats == 3 and m.parallel == 2 and m.budget_usd == 500


def test_defaults(tmp_path):
    text = GOOD.replace("parallel = 2\n", "").replace("run_timeout_minutes = 45\n", "").replace("budget_usd = 500\n", "")
    m = load_matrix(write(tmp_path, text))
    assert (m.parallel, m.run_timeout_minutes, m.budget_usd) == (1, 45, 500.0)


@pytest.mark.parametrize("bad, message", [
    (GOOD.replace('efforts = ["low", "xhigh"]', 'efforts = ["low", "turbo"]'), "unknown effort"),
    (GOOD.replace("repeats = 3", "repeats = 0"), "repeats"),
    (GOOD.replace('study_id = "demo"\n', ""), "study_id"),
    (GOOD.replace('models = ["claude-fable-5", "claude-fable-5-1"]', "models = []"), "models"),
    (GOOD.replace("parallel = 2", "parallel = 0"), "parallel"),
])
def test_rejects_bad(tmp_path, bad, message):
    with pytest.raises(MatrixError, match=message):
        load_matrix(write(tmp_path, bad))


def test_run_id_format():
    assert make_run_id("claude-fable-5", "low", "ledger-bugfix", 2) == "claude-fable-5__low__ledger-bugfix__r2"


def test_expand_is_full_grid_and_stable(tmp_path):
    m = load_matrix(write(tmp_path, GOOD))
    runs = expand_runs(m)
    assert len(runs) == 2 * 2 * 2 * 3
    assert len({r.run_id for r in runs}) == len(runs)
    assert [r.order_index for r in runs] == list(range(len(runs)))
    again = expand_runs(m)
    assert [r.run_id for r in again] == [r.run_id for r in runs]
    sorted_ids = sorted(r.run_id for r in runs)
    assert [r.run_id for r in runs] != sorted_ids  # shuffled
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_matrix.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

`bench/matrix.py`:
```python
"""Load and validate a study matrix (TOML) and expand it into shuffled run specs."""
from __future__ import annotations

import random
import tomllib
from dataclasses import dataclass
from pathlib import Path

VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")


class MatrixError(ValueError):
    """The matrix file is missing something or contains a bad value."""


@dataclass(frozen=True)
class Matrix:
    study_id: str
    claude_code_version: str
    models: tuple[str, ...]
    efforts: tuple[str, ...]
    tasks: tuple[str, ...]
    repeats: int
    parallel: int
    run_timeout_minutes: int
    budget_usd: float
    shuffle_seed: int


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    model: str
    effort: str
    task: str
    repeat: int
    order_index: int


def make_run_id(model: str, effort: str, task: str, repeat: int) -> str:
    return f"{model}__{effort}__{task}__r{repeat}"


def _str_list(data: dict, key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        raise MatrixError(f"'{key}' must be a non-empty list of strings")
    return tuple(value)


def _positive_int(data: dict, key: str, default: int | None = None) -> int:
    value = data.get(key, default)
    if value is None:
        raise MatrixError(f"matrix is missing required field '{key}'")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MatrixError(f"'{key}' must be a whole number of at least 1")
    return value


def load_matrix(path: Path) -> Matrix:
    data = tomllib.loads(Path(path).read_text())
    for key in ("study_id", "claude_code_version"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise MatrixError(f"matrix is missing required text field '{key}'")
    efforts = _str_list(data, "efforts")
    bad = [e for e in efforts if e not in VALID_EFFORTS]
    if bad:
        raise MatrixError(f"unknown effort level(s) {bad}; allowed: {', '.join(VALID_EFFORTS)}")
    budget = data.get("budget_usd", 500)
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0:
        raise MatrixError("'budget_usd' must be a positive number")
    if "shuffle_seed" not in data or isinstance(data["shuffle_seed"], bool) or not isinstance(data["shuffle_seed"], int):
        raise MatrixError("matrix is missing required whole-number field 'shuffle_seed'")
    return Matrix(
        study_id=data["study_id"],
        claude_code_version=data["claude_code_version"],
        models=_str_list(data, "models"),
        efforts=efforts,
        tasks=_str_list(data, "tasks"),
        repeats=_positive_int(data, "repeats"),
        parallel=_positive_int(data, "parallel", 1),
        run_timeout_minutes=_positive_int(data, "run_timeout_minutes", 45),
        budget_usd=float(budget),
        shuffle_seed=data["shuffle_seed"],
    )


def expand_runs(m: Matrix) -> list[RunSpec]:
    grid = [(model, effort, task, r)
            for model in m.models for effort in m.efforts
            for task in m.tasks for r in range(1, m.repeats + 1)]
    random.Random(m.shuffle_seed).shuffle(grid)
    return [RunSpec(make_run_id(*g), g[0], g[1], g[2], g[3], i) for i, g in enumerate(grid)]
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_matrix.py -q`
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add bench/matrix.py tests/test_matrix.py
git commit -m "feat: study matrix loading and grid expansion

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Tasks, fingerprints, fresh work folders (`bench/tasks.py`)

**Files:**
- Create: `bench/tasks.py`, `tests/conftest.py`, `tests/test_tasks.py`
- Create fixture task: `tests/fixtures/tasks/toy-task/task.toml`, `.../PROMPT.md`, `.../template/calc.py`, `.../template/tests/test_calc.py`, `.../hidden_tests/test_hidden_calc.py`

**Interfaces:**
- Produces: `Task` frozen dataclass (`name, version, difficulty, expected_minutes, test_paths, root`; properties `template`, `hidden_tests`, `prompt`), `load_task(name, tasks_dir=DEFAULT_TASKS_DIR) -> Task`, `fingerprint(task) -> str`, `make_workdir(task, base=None) -> Path`, `capture_diff(workdir) -> str`, `diff_stats(patch) -> tuple[int, int]`, `TaskError`, `DEFAULT_TASKS_DIR`, `GIT_IDENTITY`.
- Test fixtures used by later tasks: `toy_task`, `fake_claude`, `bench_config`, `toy_matrix_file`.

- [ ] **Step 1: Create the toy fixture task**

`tests/fixtures/tasks/toy-task/task.toml`:
```toml
name = "toy-task"
version = "1"
difficulty = "trivial"
expected_minutes = 1
test_paths = ["tests/test_calc.py"]
```

`tests/fixtures/tasks/toy-task/PROMPT.md`:
```
Fix add() in calc.py so that it adds its two arguments. Run `python -m pytest tests` before you finish.
```

`tests/fixtures/tasks/toy-task/template/calc.py`:
```python
def add(a, b):
    return a - b
```

`tests/fixtures/tasks/toy-task/template/tests/test_calc.py`:
```python
from calc import add


def test_add_zero():
    assert add(5, 0) == 5
```

`tests/fixtures/tasks/toy-task/hidden_tests/test_hidden_calc.py`:
```python
from calc import add


def test_add_two_and_three():
    assert add(2, 3) == 5
```

- [ ] **Step 2: Write conftest with the shared fixtures**

`tests/conftest.py`:
```python
import stat
from pathlib import Path

import pytest

from bench.tasks import load_task

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_CLAUDE = Path(__file__).parent / "fake_claude" / "claude"


@pytest.fixture
def toy_task():
    return load_task("toy-task", FIXTURES / "tasks")


@pytest.fixture
def fake_claude(monkeypatch):
    """Point the harness at the fake claude executable (created in Task 5)."""
    FAKE_CLAUDE.chmod(FAKE_CLAUDE.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("BENCH_CLAUDE_BIN", str(FAKE_CLAUDE))
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "ok")
    monkeypatch.delenv("FAKE_CLAUDE_VERSION", raising=False)
    return FAKE_CLAUDE


@pytest.fixture
def bench_config(tmp_path, monkeypatch):
    cfg = tmp_path / "cc-bench-config"
    cfg.mkdir()
    (cfg / "settings.json").write_text("{}\n")
    monkeypatch.setenv("BENCH_CONFIG_DIR", str(cfg))
    return cfg


@pytest.fixture
def toy_matrix_file(tmp_path):
    p = tmp_path / "toy.toml"
    p.write_text(
        'study_id = "toy-study"\n'
        'claude_code_version = "2.1.258"\n'
        'models = ["claude-fake-a", "claude-fake-b"]\n'
        'efforts = ["low", "high"]\n'
        'tasks = ["toy-task"]\n'
        'repeats = 2\n'
        'parallel = 2\n'
        'run_timeout_minutes = 1\n'
        'budget_usd = 100\n'
        'shuffle_seed = 7\n'
    )
    return p
```

- [ ] **Step 3: Write the failing tests**

`tests/test_tasks.py`:
```python
import subprocess

import pytest

from bench.tasks import (TaskError, capture_diff, diff_stats, fingerprint,
                         load_task, make_workdir)
from tests.conftest import FIXTURES


def test_load_toy(toy_task):
    assert toy_task.name == "toy-task"
    assert toy_task.test_paths == ("tests/test_calc.py",)
    assert "Fix add()" in toy_task.prompt
    assert (toy_task.template / "calc.py").exists()
    assert (toy_task.hidden_tests / "test_hidden_calc.py").exists()


def test_missing_task_raises():
    with pytest.raises(TaskError, match="not found"):
        load_task("nope", FIXTURES / "tasks")


def test_template_must_not_contain_hidden_tests(tmp_path):
    root = tmp_path / "bad"
    (root / "template" / "hidden_tests").mkdir(parents=True)
    (root / "hidden_tests").mkdir()
    (root / "PROMPT.md").write_text("x")
    (root / "task.toml").write_text('name="bad"\nversion="1"\ndifficulty="x"\nexpected_minutes=1\ntest_paths=[]\n')
    with pytest.raises(TaskError, match="must not contain"):
        load_task("bad", tmp_path)


def test_fingerprint_stable_and_sensitive(toy_task, tmp_path):
    fp1 = fingerprint(toy_task)
    assert fp1 == fingerprint(toy_task)
    assert len(fp1) == 64
    import shutil
    copy_root = tmp_path / "tasks"
    shutil.copytree(toy_task.root, copy_root / "toy-task")
    copy = load_task("toy-task", copy_root)
    assert fingerprint(copy) == fp1
    (copy.hidden_tests / "test_hidden_calc.py").write_text("def test_x():\n    assert True\n")
    assert fingerprint(copy) != fp1


def test_make_workdir_is_fresh_git_repo_without_hidden_tests(toy_task, tmp_path):
    wd = make_workdir(toy_task, base=tmp_path)
    assert (wd / "calc.py").read_text() == "def add(a, b):\n    return a - b\n"
    assert not (wd / "hidden_tests").exists()
    assert not (wd / "_hidden_tests").exists()
    log = subprocess.run(["git", "log", "--oneline"], cwd=wd, capture_output=True, text=True).stdout
    assert log.strip().endswith("baseline")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=wd, capture_output=True, text=True).stdout
    assert status == ""


def test_capture_diff_and_stats(toy_task, tmp_path):
    wd = make_workdir(toy_task, base=tmp_path)
    assert capture_diff(wd) == ""
    (wd / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (wd / "new.py").write_text("x = 1\n")
    patch = capture_diff(wd)
    assert "diff --git a/calc.py" in patch and "diff --git a/new.py" in patch
    files, lines = diff_stats(patch)
    assert files == 2
    assert lines == 3  # one removed, one added in calc.py; one added in new.py
```

- [ ] **Step 4: Run to verify failure**

Run: `python -m pytest tests/test_tasks.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.tasks'`.

- [ ] **Step 5: Write the implementation**

`bench/tasks.py`:
```python
"""Task discovery, content fingerprints, fresh work folders, and diff capture."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"
FINGERPRINT_PARTS = ("template", "hidden_tests", "PROMPT.md")
FORBIDDEN_IN_TEMPLATE = ("hidden_tests", "CLAUDE.md", ".claude")
GIT_IDENTITY = ["-c", "user.name=bench", "-c", "user.email=bench@example.com"]
WORKDIR_GITIGNORE = "__pycache__/\n.pytest_cache/\n*.pyc\n"


class TaskError(ValueError):
    """A task folder is missing or malformed."""


@dataclass(frozen=True)
class Task:
    name: str
    version: str
    difficulty: str
    expected_minutes: int
    test_paths: tuple[str, ...]
    root: Path

    @property
    def template(self) -> Path:
        return self.root / "template"

    @property
    def hidden_tests(self) -> Path:
        return self.root / "hidden_tests"

    @property
    def prompt(self) -> str:
        return (self.root / "PROMPT.md").read_text()


def load_task(name: str, tasks_dir: Path = DEFAULT_TASKS_DIR) -> Task:
    root = Path(tasks_dir) / name
    if not (root / "task.toml").exists():
        raise TaskError(f"task '{name}' not found under {tasks_dir}")
    data = tomllib.loads((root / "task.toml").read_text())
    for key in ("name", "version", "difficulty", "expected_minutes", "test_paths"):
        if key not in data:
            raise TaskError(f"task '{name}': task.toml is missing '{key}'")
    for sub in ("template", "hidden_tests"):
        if not (root / sub).is_dir():
            raise TaskError(f"task '{name}': missing folder {sub}/")
    if not (root / "PROMPT.md").exists():
        raise TaskError(f"task '{name}': missing PROMPT.md")
    for bad in FORBIDDEN_IN_TEMPLATE:
        if (root / "template" / bad).exists():
            raise TaskError(f"task '{name}': template must not contain {bad}")
    return Task(
        name=str(data["name"]),
        version=str(data["version"]),
        difficulty=str(data["difficulty"]),
        expected_minutes=int(data["expected_minutes"]),
        test_paths=tuple(str(p) for p in data["test_paths"]),
        root=root,
    )


def _files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(f for f in path.rglob("*")
                  if f.is_file() and "__pycache__" not in f.parts and ".pytest_cache" not in f.parts)


def fingerprint(task: Task) -> str:
    """SHA-256 over relative path + content of template/, hidden_tests/ and PROMPT.md."""
    h = hashlib.sha256()
    for part in FINGERPRINT_PARTS:
        for f in _files_under(task.root / part):
            h.update(f.relative_to(task.root).as_posix().encode())
            h.update(b"\0")
            h.update(f.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def _git(workdir: Path, *args: str) -> str:
    result = subprocess.run(["git", *GIT_IDENTITY, *args], cwd=workdir,
                            check=True, capture_output=True, text=True)
    return result.stdout


def make_workdir(task: Task, base: Path | None = None) -> Path:
    """Fresh temp folder holding a copy of the template, committed as 'baseline'."""
    workdir = Path(tempfile.mkdtemp(prefix=f"ccbench-{task.name}-", dir=base))
    shutil.copytree(task.template, workdir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    (workdir / ".gitignore").write_text(WORKDIR_GITIGNORE)
    _git(workdir, "init", "-q", "-b", "main")
    _git(workdir, "add", "-A")
    _git(workdir, "commit", "-q", "-m", "baseline")
    return workdir


def capture_diff(workdir: Path) -> str:
    """Stage everything and return the diff against the baseline commit."""
    _git(workdir, "add", "-A")
    return _git(workdir, "diff", "--cached", "--binary")


def diff_stats(patch: str) -> tuple[int, int]:
    files = sum(1 for line in patch.splitlines() if line.startswith("diff --git "))
    changed = sum(1 for line in patch.splitlines()
                  if (line.startswith("+") and not line.startswith("+++"))
                  or (line.startswith("-") and not line.startswith("---")))
    return files, changed
```

- [ ] **Step 6: Run to verify pass**

Run: `python -m pytest tests/test_tasks.py -q`
Expected: `6 passed`.

- [ ] **Step 7: Commit**

```bash
git add bench/tasks.py tests/conftest.py tests/test_tasks.py tests/fixtures
git commit -m "feat: task loading, fingerprints, fresh work folders

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Claude Code command line + event parsing + fake claude (`bench/claude_cli.py`)

**Files:**
- Create: `bench/claude_cli.py`, `tests/fake_claude/claude`, `tests/test_claude_cli.py`

**Interfaces:**
- Produces: `claude_bin() -> str`, `config_dir() -> Path`, `build_env() -> dict`, `build_run_command(model, effort, prompt) -> list[str]`, `build_probe_command(model) -> list[str]`, `claude_version() -> str`, `ParsedStream` (`init, result, rate_limit_events, malformed_lines`), `parse_stream(text) -> ParsedStream`, `utilization_5h(parsed) -> float | None`, `classify(parsed, timed_out) -> tuple[str, str | None]`, `usage_fields(result) -> dict`, `PROBE_PROMPT`.
- Fake claude honours env `FAKE_CLAUDE_SCENARIO` in {`ok`, `wrong`, `edits_tests`, `rate_limited`, `api_error`, `no_result`, `hang`}, `FAKE_CLAUDE_VERSION` (default `2.1.258`), `FAKE_CLAUDE_HANG_S` (default `30`). It writes `calc.py` in its current folder for `ok`/`wrong`/`edits_tests`.

- [ ] **Step 1: Write the fake claude executable**

`tests/fake_claude/claude`:
```python
#!/usr/bin/env python3
"""Fake `claude` for harness tests. Never calls any API. Mimics the real
stream-json and json output shapes observed on Claude Code 2.1.258."""
import json
import os
import sys
import time
from pathlib import Path

VERSION = os.environ.get("FAKE_CLAUDE_VERSION", "2.1.258")
SCENARIO = os.environ.get("FAKE_CLAUDE_SCENARIO", "ok")


def usage():
    return {"input_tokens": 10, "cache_creation_input_tokens": 1000,
            "cache_read_input_tokens": 2000, "output_tokens": 300,
            "output_tokens_details": {"thinking_tokens": 100}}


def result_event(model, is_error=False, text="DONE", terminal="completed", cost=0.05):
    return {"type": "result", "subtype": "success", "is_error": is_error, "result": text,
            "terminal_reason": terminal, "num_turns": 3, "duration_ms": 1200,
            "duration_api_ms": 1500, "total_cost_usd": cost, "usage": usage(),
            "modelUsage": {model: {"costBasis": "list", "costUSD": cost}}}


def main(argv):
    if "--version" in argv:
        print(f"{VERSION} (Claude Code)")
        return 0

    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    model = opt("--model", "claude-fake")
    if opt("--output-format", "text") == "json":
        print(json.dumps(result_event(model, text="OK", cost=0.01)))
        return 0

    def emit(ev):
        print(json.dumps(ev), flush=True)

    emit({"type": "system", "subtype": "init", "cwd": os.getcwd(), "model": model,
          "permissionMode": opt("--permission-mode"), "mcp_servers": []})
    status = "rejected" if SCENARIO == "rate_limited" else "allowed"
    emit({"type": "rate_limit_event", "rate_limit_info": {
        "status": status, "rateLimitType": "five_hour", "resetsAt": 1788321600,
        "unifiedWindows": {"five_hour": {"utilization": 0.42, "resetsAt": 1788321600}}}})
    if SCENARIO == "hang":
        time.sleep(float(os.environ.get("FAKE_CLAUDE_HANG_S", "30")))
    if SCENARIO == "rate_limited":
        emit(result_event(model, is_error=True, text="You've hit your usage limit. Resets 3pm.", terminal="api_error"))
        return 1
    if SCENARIO == "api_error":
        emit(result_event(model, is_error=True, text="API Error: 500 internal", terminal="api_error"))
        return 1
    if SCENARIO == "no_result":
        return 1
    cwd = Path.cwd()
    if SCENARIO in ("ok", "edits_tests"):
        (cwd / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    elif SCENARIO == "wrong":
        (cwd / "calc.py").write_text("def add(a, b):\n    return a * b\n")
    if SCENARIO == "edits_tests":
        (cwd / "tests" / "test_calc.py").write_text("def test_nothing():\n    assert True\n")
    emit({"type": "assistant", "message": {"content": [{"type": "text", "text": "DONE"}]}})
    emit(result_event(model))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

Then: `chmod +x tests/fake_claude/claude`.

- [ ] **Step 2: Write the failing tests**

`tests/test_claude_cli.py`:
```python
import json
import subprocess

from bench.claude_cli import (PROBE_PROMPT, build_env, build_probe_command,
                              build_run_command, claude_version, classify,
                              parse_stream, usage_fields, utilization_5h)


def test_run_command_shape(fake_claude, bench_config):
    cmd = build_run_command("claude-fable-5-1", "low", "do the thing")
    assert cmd[0] == str(fake_claude)
    assert cmd[1:3] == ["-p", "do the thing"]
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "claude-fable-5-1"
    assert cmd[cmd.index("--effort") + 1] == "low"
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    for flag in ("--verbose", "--strict-mcp-config", "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"


def test_probe_command_uses_json(fake_claude, bench_config):
    cmd = build_probe_command("claude-haiku-4-5")
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert PROBE_PROMPT in cmd


def test_env_sets_config_dir(fake_claude, bench_config):
    assert build_env()["CLAUDE_CONFIG_DIR"] == str(bench_config)


def test_claude_version(fake_claude, bench_config, monkeypatch):
    assert claude_version() == "2.1.258"
    monkeypatch.setenv("FAKE_CLAUDE_VERSION", "9.9.9")
    assert claude_version() == "9.9.9"


def run_fake(fake_claude, bench_config, tmp_path, scenario, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", scenario)
    cmd = build_run_command("claude-fake", "low", "x")
    proc = subprocess.run(cmd, cwd=tmp_path, env=build_env(), capture_output=True, text=True)
    return parse_stream(proc.stdout)


def test_parse_ok_stream(fake_claude, bench_config, tmp_path, monkeypatch):
    parsed = run_fake(fake_claude, bench_config, tmp_path, "ok", monkeypatch)
    assert parsed.init["subtype"] == "init"
    assert parsed.result["num_turns"] == 3
    assert utilization_5h(parsed) == 0.42
    assert classify(parsed, timed_out=False) == ("completed", None)
    fields = usage_fields(parsed.result)
    assert fields["input_tokens"] == 10 and fields["thinking_tokens"] == 100
    assert fields["cost_basis"] == "list" and fields["total_cost_usd"] == 0.05


def test_classify_rate_limited_event(fake_claude, bench_config, tmp_path, monkeypatch):
    parsed = run_fake(fake_claude, bench_config, tmp_path, "rate_limited", monkeypatch)
    status, text = classify(parsed, timed_out=False)
    assert status == "rate_limited" and "rejected" in text


def test_classify_rate_limit_by_text():
    stream = json.dumps({"type": "result", "is_error": True, "result": "You've hit your usage limit"})
    assert classify(parse_stream(stream), timed_out=False)[0] == "rate_limited"


def test_classify_api_error(fake_claude, bench_config, tmp_path, monkeypatch):
    parsed = run_fake(fake_claude, bench_config, tmp_path, "api_error", monkeypatch)
    assert classify(parsed, timed_out=False)[0] == "api_error"


def test_classify_no_result_and_timeout(fake_claude, bench_config, tmp_path, monkeypatch):
    parsed = run_fake(fake_claude, bench_config, tmp_path, "no_result", monkeypatch)
    assert classify(parsed, timed_out=False) == ("api_error", "no result event in stream")
    assert classify(parsed, timed_out=True)[0] == "timeout"


def test_parse_skips_malformed_lines():
    parsed = parse_stream('not json\n{"type": "result", "is_error": false, "terminal_reason": "completed"}\n')
    assert parsed.malformed_lines == 1 and parsed.result is not None
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_claude_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.claude_cli'`.

- [ ] **Step 4: Write the implementation**

`bench/claude_cli.py`:
```python
"""Everything that touches the `claude` command line: building it, running
--version, parsing the stream-json events, and classifying the outcome."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

PROBE_PROMPT = "Reply with exactly the word OK and nothing else."
RATE_LIMIT_WORDS = ("usage limit", "rate limit", "hit your limit", "limit reached",
                    "out of extra usage", "limit will reset")


def claude_bin() -> str:
    return os.environ.get("BENCH_CLAUDE_BIN", "claude")


def config_dir() -> Path:
    return Path(os.environ.get("BENCH_CONFIG_DIR", str(Path.home() / ".cc-bench"))).expanduser()


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    env["CLAUDE_CONFIG_DIR"] = str(config_dir())
    return env


def build_run_command(model: str, effort: str, prompt: str) -> list[str]:
    return [claude_bin(), "-p", prompt,
            "--model", model, "--effort", effort,
            "--output-format", "stream-json", "--verbose",
            "--permission-mode", "bypassPermissions",
            "--strict-mcp-config", "--no-session-persistence"]


def build_probe_command(model: str) -> list[str]:
    return [claude_bin(), "-p", PROBE_PROMPT, "--model", model,
            "--output-format", "json", "--strict-mcp-config", "--no-session-persistence"]


def claude_version() -> str:
    result = subprocess.run([claude_bin(), "--version"], capture_output=True,
                            text=True, env=build_env(), timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"'{claude_bin()} --version' failed: {result.stderr.strip()}")
    return result.stdout.strip().split()[0]


@dataclass
class ParsedStream:
    init: dict | None = None
    result: dict | None = None
    rate_limit_events: list[dict] = field(default_factory=list)
    malformed_lines: int = 0


def parse_stream(text: str) -> ParsedStream:
    parsed = ParsedStream()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            parsed.malformed_lines += 1
            continue
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            parsed.init = event
        elif kind == "result":
            parsed.result = event
        elif kind == "rate_limit_event":
            parsed.rate_limit_events.append(event)
    return parsed


def utilization_5h(parsed: ParsedStream) -> float | None:
    for event in reversed(parsed.rate_limit_events):
        try:
            return float(event["rate_limit_info"]["unifiedWindows"]["five_hour"]["utilization"])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def classify(parsed: ParsedStream, timed_out: bool) -> tuple[str, str | None]:
    """Return (status, error_text). Status is one of bench.schema.STATUSES."""
    if timed_out:
        return "timeout", "killed by the runner after the run timeout"
    for event in parsed.rate_limit_events:
        status = (event.get("rate_limit_info") or {}).get("status")
        if status and status != "allowed":
            return "rate_limited", f"rate_limit_event status={status}"
    result = parsed.result
    if result is None:
        return "api_error", "no result event in stream"
    text = str(result.get("result") or result.get("error") or "")
    if result.get("is_error"):
        if any(word in text.lower() for word in RATE_LIMIT_WORDS):
            return "rate_limited", text[:500]
        return "api_error", text[:500] or "result is_error=true"
    reason = result.get("terminal_reason")
    if reason not in (None, "completed"):
        return "api_error", f"terminal_reason={reason}"
    return "completed", None


def usage_fields(result: dict) -> dict:
    """Pick the usage numbers out of a result event, keyed like RunMeta fields."""
    usage = result.get("usage") or {}
    details = usage.get("output_tokens_details") or {}
    cost_basis = None
    for per_model in (result.get("modelUsage") or {}).values():
        cost_basis = per_model.get("costBasis")
        break
    return dict(
        duration_api_ms=result.get("duration_api_ms"),
        terminal_reason=result.get("terminal_reason"),
        num_turns=result.get("num_turns"),
        input_tokens=usage.get("input_tokens"),
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
        cache_read_input_tokens=usage.get("cache_read_input_tokens"),
        output_tokens=usage.get("output_tokens"),
        thinking_tokens=details.get("thinking_tokens"),
        total_cost_usd=result.get("total_cost_usd"),
        cost_basis=cost_basis,
    )
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_claude_cli.py -q`
Expected: `10 passed`.

- [ ] **Step 6: Commit**

```bash
git add bench/claude_cli.py tests/fake_claude/claude tests/test_claude_cli.py
git commit -m "feat: claude command builder, stream parser, fake claude for tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Scorer (`bench/scorer.py`)

**Files:**
- Create: `bench/scorer.py`, `tests/test_scorer.py`

**Interfaces:**
- Consumes: `Task`, `make_workdir`, `GIT_IDENTITY` from `bench.tasks`; `RunMeta` from `bench.schema`.
- Produces: `parse_junit(path) -> tuple[int, int]`, `run_hidden_tests(workdir, task, junit_path) -> tuple[int, int]`, `visible_tests_modified(task, workdir) -> bool`, `score_run(meta, workdir, task, run_dir) -> RunMeta`, `rebuild_workdir(task, patch_path) -> Path`, `score_study(matrix, results_root, tasks_dir, log=print) -> int`, constants `HIDDEN_DIRNAME = "_hidden_tests"`, `TEST_TIMEOUT_S = 120`.

- [ ] **Step 1: Write the failing tests**

`tests/test_scorer.py`:
```python
from pathlib import Path

from bench.scorer import (HIDDEN_DIRNAME, parse_junit, rebuild_workdir,
                          run_hidden_tests, score_run, visible_tests_modified)
from bench.tasks import capture_diff, make_workdir
from tests.test_schema import make_meta


def fix(wd: Path, body: str = "return a + b"):
    (wd / "calc.py").write_text(f"def add(a, b):\n    {body}\n")


def test_parse_junit(tmp_path):
    x = tmp_path / "j.xml"
    x.write_text('<testsuites><testsuite tests="4" failures="1" errors="1" skipped="0"/></testsuites>')
    assert parse_junit(x) == (4, 2)


def test_hidden_tests_pass_after_fix(toy_task, tmp_path):
    wd = make_workdir(toy_task, base=tmp_path)
    fix(wd)
    assert run_hidden_tests(wd, toy_task, tmp_path / "j.xml") == (1, 1)
    assert (wd / HIDDEN_DIRNAME / "test_hidden_calc.py").exists()


def test_hidden_tests_fail_on_template(toy_task, tmp_path):
    wd = make_workdir(toy_task, base=tmp_path)
    assert run_hidden_tests(wd, toy_task, tmp_path / "j.xml") == (1, 0)


def test_visible_tests_modified(toy_task, tmp_path):
    wd = make_workdir(toy_task, base=tmp_path)
    assert visible_tests_modified(toy_task, wd) is False
    (wd / "tests" / "test_calc.py").write_text("def test_x():\n    assert True\n")
    assert visible_tests_modified(toy_task, wd) is True
    (wd / "tests" / "test_calc.py").unlink()
    assert visible_tests_modified(toy_task, wd) is True


def test_score_run_fills_meta(toy_task, tmp_path):
    wd = make_workdir(toy_task, base=tmp_path)
    fix(wd, "return a * b")
    rd = tmp_path / "run"
    rd.mkdir()
    meta = score_run(make_meta(status="completed"), wd, toy_task, rd)
    assert (meta.tests_total, meta.tests_passed, meta.passed) == (1, 0, False)
    assert meta.visible_tests_modified is False
    assert (rd / "junit.xml").exists()


def test_rebuild_workdir_from_patch(toy_task, tmp_path):
    wd = make_workdir(toy_task, base=tmp_path)
    fix(wd)
    (wd / "extra.txt").write_text("hello\n")
    patch = tmp_path / "diff.patch"
    patch.write_text(capture_diff(wd))
    rebuilt = rebuild_workdir(toy_task, patch)
    assert (rebuilt / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    assert (rebuilt / "extra.txt").read_text() == "hello\n"
    assert run_hidden_tests(rebuilt, toy_task, tmp_path / "j2.xml") == (1, 1)


def test_rebuild_with_empty_patch(toy_task, tmp_path):
    patch = tmp_path / "empty.patch"
    patch.write_text("")
    rebuilt = rebuild_workdir(toy_task, patch)
    assert (rebuilt / "calc.py").read_text() == "def add(a, b):\n    return a - b\n"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_scorer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.scorer'`.

- [ ] **Step 3: Write the implementation**

`bench/scorer.py`:
```python
"""Hidden-test scoring and the cheat check. Runs after the diff is captured."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from .matrix import Matrix
from .schema import RunMeta
from .tasks import GIT_IDENTITY, Task, load_task, make_workdir

HIDDEN_DIRNAME = "_hidden_tests"
TEST_TIMEOUT_S = 120


def parse_junit(path: Path) -> tuple[int, int]:
    """(total, passed) from a pytest junit file. Skipped tests count as not passed."""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    total = passed = 0
    for suite in suites:
        tests = int(suite.get("tests", 0))
        bad = int(suite.get("failures", 0)) + int(suite.get("errors", 0)) + int(suite.get("skipped", 0))
        total += tests
        passed += tests - bad
    return total, passed


def run_hidden_tests(workdir: Path, task: Task, junit_path: Path) -> tuple[int, int]:
    dest = workdir / HIDDEN_DIRNAME
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(task.hidden_tests, dest)
    env = dict(os.environ, PYTHONPATH=str(workdir), PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "pytest", HIDDEN_DIRNAME, "-q", "-p", "no:cacheprovider",
           f"--junitxml={junit_path}"]
    try:
        subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True,
                       timeout=TEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return 0, 0
    if not Path(junit_path).exists():
        return 0, 0
    return parse_junit(Path(junit_path))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visible_tests_modified(task: Task, workdir: Path) -> bool:
    for rel in task.test_paths:
        original = task.template / rel
        current = workdir / rel
        if not current.exists() or _sha(original) != _sha(current):
            return True
    return False


def score_run(meta: RunMeta, workdir: Path, task: Task, run_dir: Path) -> RunMeta:
    total, passed = run_hidden_tests(workdir, task, Path(run_dir) / "junit.xml")
    meta.tests_total = total
    meta.tests_passed = passed
    meta.passed = total > 0 and passed == total
    meta.visible_tests_modified = visible_tests_modified(task, workdir)
    return meta


def rebuild_workdir(task: Task, patch_path: Path) -> Path:
    """Fresh template copy with the run's diff applied. Used by `bench score`."""
    workdir = make_workdir(task)
    if Path(patch_path).read_text().strip():
        subprocess.run(["git", *GIT_IDENTITY, "apply", "--index", str(Path(patch_path).resolve())],
                       cwd=workdir, check=True, capture_output=True, text=True)
    return workdir


def score_study(matrix: Matrix, results_root: Path, tasks_dir: Path, log=print) -> int:
    """Score every completed-but-unscored run. Returns how many were scored."""
    runs_root = Path(results_root) / matrix.study_id / "runs"
    if not runs_root.exists():
        return 0
    tasks: dict[str, Task] = {}
    scored = 0
    for run_dir in sorted(runs_root.iterdir()):
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = RunMeta.load(meta_path)
        if meta.status != "completed" or meta.passed is not None:
            continue
        task = tasks.setdefault(meta.task, load_task(meta.task, tasks_dir))
        workdir = rebuild_workdir(task, run_dir / "diff.patch")
        try:
            meta = score_run(meta, workdir, task, run_dir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        meta.save(meta_path)
        scored += 1
        log(f"scored {meta.run_id}: passed={meta.passed} ({meta.tests_passed}/{meta.tests_total})")
    return scored
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_scorer.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add bench/scorer.py tests/test_scorer.py
git commit -m "feat: hidden-test scorer with cheat check and rebuild-from-patch

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Runner (`bench/runner.py`)

**Files:**
- Create: `bench/runner.py`, `tests/test_runner.py`

**Interfaces:**
- Consumes: `build_env, build_run_command, classify, parse_stream, usage_fields, utilization_5h` (Task 5); `Matrix, RunSpec, expand_runs` (Task 3); `RunMeta` (Task 2); `score_run` (Task 6); `Task, capture_diff, diff_stats, fingerprint, load_task, make_workdir` (Task 4).
- Produces: `study_dir(results_root, study_id) -> Path`, `run_dir(results_root, study_id, run_id) -> Path`, `is_completed(run_dir) -> bool`, `execute_run(spec, matrix, task, task_fp, results_root, cc_version, timeout_s, keep_workdir=False) -> RunMeta`, `run_study(matrix, results_root, tasks_dir, cc_version, parallel=None, only=None, timeout_s=None, log=print) -> StudyOutcome`, `StudyOutcome(executed, skipped, spent_usd, stop_reason)`, `format_line(meta) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_runner.py`:
```python
import json

import pytest

from bench.matrix import load_matrix
from bench.runner import execute_run, is_completed, run_dir, run_study
from bench.schema import RunMeta
from bench.tasks import fingerprint
from tests.conftest import FIXTURES

TASKS = FIXTURES / "tasks"


def spec_for(matrix, index=0):
    from bench.matrix import expand_runs
    return expand_runs(matrix)[index]


def do_run(matrix, toy_task, tmp_path, timeout_s=30, keep=False):
    spec = spec_for(matrix)
    return spec, execute_run(spec, matrix, toy_task, fingerprint(toy_task), tmp_path / "results",
                             "2.1.258", timeout_s=timeout_s, keep_workdir=keep)


def test_ok_run_is_completed_and_scored(fake_claude, bench_config, toy_task, toy_matrix_file, tmp_path):
    matrix = load_matrix(toy_matrix_file)
    spec, meta = do_run(matrix, toy_task, tmp_path)
    rd = run_dir(tmp_path / "results", "toy-study", spec.run_id)
    assert meta.status == "completed" and meta.passed is True
    assert meta.tokens_total == 3310 and meta.thinking_tokens == 100
    assert meta.total_cost_usd == 0.05 and meta.cost_basis == "list"
    assert meta.rate_limit_utilization_5h == 0.42
    assert meta.diff_files_changed == 1 and meta.diff_lines == 2
    assert meta.visible_tests_modified is False
    assert meta.started_at and meta.ended_at and meta.duration_ms >= 0
    for name in ("meta.json", "stream.jsonl", "result.json", "diff.patch", "stderr.log", "junit.xml"):
        assert (rd / name).exists(), name
    assert RunMeta.load(rd / "meta.json") == meta
    assert is_completed(rd)
    init = json.loads((rd / "stream.jsonl").read_text().splitlines()[0])
    assert "ccbench-toy-task-" in init["cwd"]


def test_wrong_fix_fails_hidden_tests(fake_claude, bench_config, toy_task, toy_matrix_file, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "wrong")
    _, meta = do_run(load_matrix(toy_matrix_file), toy_task, tmp_path)
    assert meta.status == "completed" and meta.passed is False
    assert (meta.tests_total, meta.tests_passed) == (1, 0)


def test_editing_visible_tests_is_flagged(fake_claude, bench_config, toy_task, toy_matrix_file, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "edits_tests")
    _, meta = do_run(load_matrix(toy_matrix_file), toy_task, tmp_path)
    assert meta.passed is True and meta.visible_tests_modified is True


def test_timeout_kills_and_records(fake_claude, bench_config, toy_task, toy_matrix_file, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "hang")
    monkeypatch.setenv("FAKE_CLAUDE_HANG_S", "20")
    _, meta = do_run(load_matrix(toy_matrix_file), toy_task, tmp_path, timeout_s=1)
    assert meta.status == "timeout" and meta.duration_ms < 10_000
    assert meta.passed is None


def test_api_error_recorded(fake_claude, bench_config, toy_task, toy_matrix_file, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "api_error")
    spec, meta = do_run(load_matrix(toy_matrix_file), toy_task, tmp_path)
    assert meta.status == "api_error" and "500" in meta.error_text
    assert (run_dir(tmp_path / "results", "toy-study", spec.run_id) / "meta.json").exists()


def test_rate_limited_run_is_moved_aside(fake_claude, bench_config, toy_task, toy_matrix_file, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "rate_limited")
    spec, meta = do_run(load_matrix(toy_matrix_file), toy_task, tmp_path)
    assert meta.status == "rate_limited"
    assert not run_dir(tmp_path / "results", "toy-study", spec.run_id).exists()
    aside = list((tmp_path / "results" / "toy-study" / "rate_limited").iterdir())
    assert len(aside) == 1 and (aside[0] / "stream.jsonl").exists()


def test_run_study_then_resume_skips(fake_claude, bench_config, toy_matrix_file, tmp_path):
    matrix = load_matrix(toy_matrix_file)
    first = run_study(matrix, tmp_path / "results", TASKS, "2.1.258", log=lambda s: None)
    assert (first.executed, first.skipped, first.stop_reason) == (8, 0, None)
    assert first.spent_usd == pytest.approx(0.40)
    second = run_study(matrix, tmp_path / "results", TASKS, "2.1.258", log=lambda s: None)
    assert (second.executed, second.skipped) == (0, 8)
    assert second.spent_usd == pytest.approx(0.40)


def test_run_study_stops_on_rate_limit(fake_claude, bench_config, toy_matrix_file, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "rate_limited")
    matrix = load_matrix(toy_matrix_file)
    out = run_study(matrix, tmp_path / "results", TASKS, "2.1.258", parallel=1, log=lambda s: None)
    assert out.executed == 1 and "rate limited" in out.stop_reason
    assert not (tmp_path / "results" / "toy-study" / "runs").exists() or \
        not any((tmp_path / "results" / "toy-study" / "runs").iterdir())


def test_run_study_stops_on_budget(fake_claude, bench_config, toy_matrix_file, tmp_path):
    toy_matrix_file.write_text(toy_matrix_file.read_text().replace("budget_usd = 100", "budget_usd = 0.06"))
    matrix = load_matrix(toy_matrix_file)
    out = run_study(matrix, tmp_path / "results", TASKS, "2.1.258", parallel=1, log=lambda s: None)
    assert out.executed == 2 and "budget cap" in out.stop_reason


def test_run_study_only_reruns_one(fake_claude, bench_config, toy_matrix_file, tmp_path):
    matrix = load_matrix(toy_matrix_file)
    run_study(matrix, tmp_path / "results", TASKS, "2.1.258", log=lambda s: None)
    target = spec_for(matrix, 3).run_id
    out = run_study(matrix, tmp_path / "results", TASKS, "2.1.258", only=target, log=lambda s: None)
    assert (out.executed, out.skipped) == (1, 0)
    with pytest.raises(ValueError, match="no run id"):
        run_study(matrix, tmp_path / "results", TASKS, "2.1.258", only="nope", log=lambda s: None)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_runner.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.runner'`.

- [ ] **Step 3: Write the implementation**

`bench/runner.py`:
```python
"""Run one cell of the matrix: fresh work folder, claude -p, capture, classify, score."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .claude_cli import (build_env, build_run_command, classify, parse_stream,
                         usage_fields, utilization_5h)
from .matrix import Matrix, RunSpec, expand_runs
from .schema import RunMeta
from .scorer import score_run
from .tasks import Task, capture_diff, diff_stats, fingerprint, load_task, make_workdir


@dataclass
class StudyOutcome:
    executed: int
    skipped: int
    spent_usd: float
    stop_reason: str | None


def study_dir(results_root: Path, study_id: str) -> Path:
    return Path(results_root) / study_id


def run_dir(results_root: Path, study_id: str, run_id: str) -> Path:
    return study_dir(results_root, study_id) / "runs" / run_id


def is_completed(rd: Path) -> bool:
    meta_path = Path(rd) / "meta.json"
    if not meta_path.exists():
        return False
    try:
        return RunMeta.load(meta_path).status == "completed"
    except (ValueError, TypeError, KeyError):
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_line(meta: RunMeta) -> str:
    cost = f"${meta.total_cost_usd:.2f}" if meta.total_cost_usd is not None else "$-"
    util = f"{meta.rate_limit_utilization_5h:.0%}" if meta.rate_limit_utilization_5h is not None else "-"
    return (f"{meta.run_id:<58} {meta.status:<13} tokens={meta.tokens_total or '-':<8} "
            f"cost={cost:<7} turns={meta.num_turns or '-':<4} min={meta.duration_ms / 60000:5.1f} "
            f"pass={meta.passed} util5h={util}")


def execute_run(spec: RunSpec, matrix: Matrix, task: Task, task_fp: str, results_root: Path,
                cc_version: str, timeout_s: float, keep_workdir: bool = False) -> RunMeta:
    rd = run_dir(results_root, matrix.study_id, spec.run_id)
    if rd.exists():
        shutil.rmtree(rd)
    rd.mkdir(parents=True)
    meta = RunMeta(study_id=matrix.study_id, run_id=spec.run_id, model=spec.model,
                   effort=spec.effort, task=task.name, task_version=task.version,
                   task_fingerprint=task_fp, repeat=spec.repeat, order_index=spec.order_index,
                   shuffle_seed=matrix.shuffle_seed, claude_code_version=cc_version)
    workdir: Path | None = None
    try:
        workdir = make_workdir(task)
        command = build_run_command(spec.model, spec.effort, task.prompt)
        meta.started_at = _now()
        started = time.monotonic()
        timed_out = False
        with open(rd / "stream.jsonl", "wb") as out, open(rd / "stderr.log", "wb") as err:
            proc = subprocess.Popen(command, cwd=workdir, env=build_env(), stdout=out, stderr=err,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
        meta.duration_ms = int((time.monotonic() - started) * 1000)
        meta.ended_at = _now()
        parsed = parse_stream((rd / "stream.jsonl").read_text(errors="replace"))
        if parsed.result is not None:
            (rd / "result.json").write_text(json.dumps(parsed.result, indent=2))
            for key, value in usage_fields(parsed.result).items():
                setattr(meta, key, value)
        patch = capture_diff(workdir)
        (rd / "diff.patch").write_text(patch)
        meta.diff_files_changed, meta.diff_lines = diff_stats(patch)
        meta.status, meta.error_text = classify(parsed, timed_out)
        meta.rate_limit_utilization_5h = utilization_5h(parsed)
        if meta.status == "completed":
            meta = score_run(meta, workdir, task, rd)
    except Exception as exc:  # harness bug: record it, never crash the study
        meta.status = "harness_error"
        meta.error_text = f"{type(exc).__name__}: {exc}"
        meta.ended_at = meta.ended_at or _now()
        (rd / "traceback.txt").write_text(traceback.format_exc())
    finally:
        if workdir is not None and not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
    if meta.status == "rate_limited":
        # Keep the evidence, but free the run id so it is retried on resume.
        aside = study_dir(results_root, matrix.study_id) / "rate_limited"
        aside.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        meta.save(rd / "meta.json")
        shutil.move(str(rd), str(aside / f"{spec.run_id}-{stamp}"))
    else:
        meta.save(rd / "meta.json")
    return meta


def _existing_spend(results_root: Path, study_id: str) -> float:
    runs_root = study_dir(results_root, study_id) / "runs"
    if not runs_root.exists():
        return 0.0
    total = 0.0
    for rd in runs_root.iterdir():
        if (rd / "meta.json").exists():
            total += RunMeta.load(rd / "meta.json").total_cost_usd or 0.0
    return total


def run_study(matrix: Matrix, results_root: Path, tasks_dir: Path, cc_version: str,
              parallel: int | None = None, only: str | None = None,
              timeout_s: float | None = None, log=print) -> StudyOutcome:
    specs = expand_runs(matrix)
    if only is not None:
        specs = [s for s in specs if s.run_id == only]
        if not specs:
            raise ValueError(f"no run id '{only}' in this matrix")
    tasks = {name: load_task(name, tasks_dir) for name in matrix.tasks}
    fps = {name: fingerprint(task) for name, task in tasks.items()}
    timeout = timeout_s if timeout_s is not None else matrix.run_timeout_minutes * 60
    lock = threading.Lock()
    state = {"executed": 0, "skipped": 0, "spent": _existing_spend(results_root, matrix.study_id),
             "stop": None}

    def work(spec: RunSpec) -> None:
        with lock:
            if state["stop"]:
                return
            rd = run_dir(results_root, matrix.study_id, spec.run_id)
            if only is None and is_completed(rd):
                state["skipped"] += 1
                log(f"{spec.run_id:<58} skipped (already completed)")
                return
        meta = execute_run(spec, matrix, tasks[spec.task], fps[spec.task], results_root,
                           cc_version, timeout)
        with lock:
            state["executed"] += 1
            state["spent"] += meta.total_cost_usd or 0.0
            if meta.status == "rate_limited" and not state["stop"]:
                state["stop"] = f"rate limited: {meta.error_text}"
            if state["spent"] > matrix.budget_usd and not state["stop"]:
                state["stop"] = (f"budget cap ${matrix.budget_usd:.2f} exceeded "
                                 f"(spent ${state['spent']:.2f} list price)")
        log(format_line(meta))

    with ThreadPoolExecutor(max_workers=parallel or matrix.parallel) as pool:
        list(pool.map(work, specs))
    return StudyOutcome(state["executed"], state["skipped"], state["spent"], state["stop"])
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_runner.py -q`
Expected: `10 passed` (the timeout test takes about one second; the full-study tests a few seconds each).

- [ ] **Step 5: Commit**

```bash
git add bench/runner.py tests/test_runner.py
git commit -m "feat: runner with fresh work folders, resume, rate-limit and budget stops

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Report (`bench/report.py`)

**Files:**
- Create: `bench/report.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: `RunMeta` (Task 2); `results/<study>/environment.json` and `results/<study>/baseline/<model>.json` (written by Task 9; shapes given below).
- Produces: `CellSummary` dataclass, `load_runs(results_root, study_id) -> list[RunMeta]`, `load_baselines(results_root, study_id) -> dict[str, dict]`, `load_environment(results_root, study_id) -> dict`, `summarize(runs, baselines) -> list[CellSummary]`, `write_csv(cells, path)`, `write_runs_csv(runs, path)`, `write_markdown(cells, envs, baselines, path)`, `write_chart(cells, path) -> bool`, `report_study(results_root, study_ids, log=print) -> Path`, `ReportError`.
- File shapes this task relies on:
  - `environment.json`: `{"study_id", "claude_code_version", "recorded_at", "python", "os", "matrix": {...}, "task_fingerprints": {task: fp}}`
  - `baseline/<model>.json`: `{"model", "recorded_at", "total_cost_usd", "usage": {...}}`

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:
```python
import csv
import json

import pytest

from bench.report import (ReportError, report_study, summarize, write_chart,
                          write_csv, write_markdown)
from tests.test_schema import make_meta


def run(model, effort, task, status="completed", passed=True, cost=1.0, **kw):
    return make_meta(model=model, effort=effort, task=task, run_id=f"{model}__{effort}__{task}__r{kw.pop('r', 1)}",
                     status=status, passed=passed if status == "completed" else None,
                     total_cost_usd=cost, input_tokens=10, cache_creation_input_tokens=100,
                     cache_read_input_tokens=1000, output_tokens=50, thinking_tokens=20,
                     num_turns=4, duration_ms=120_000, visible_tests_modified=False, **kw)


def test_summarize_cell_math():
    runs = [run("m", "low", "t", cost=1.0, r=1), run("m", "low", "t", cost=3.0, passed=False, r=2),
            run("m", "low", "t", cost=2.0, r=3), run("m", "low", "t", status="timeout", r=4)]
    [cell] = summarize(runs, {})
    assert (cell.n_completed, cell.n_pass, cell.n_other) == (3, 2, 1)
    assert cell.pass_rate == pytest.approx(2 / 3)
    assert cell.cost_median == 2.0 and (cell.cost_min, cell.cost_max) == (1.0, 3.0)
    assert cell.cost_per_pass == pytest.approx(3.0)  # 6.0 total / 2 passes
    assert cell.tokens_total_median == 1160 and cell.turns_median == 4 and cell.minutes_median == 2.0
    assert cell.baseline_cost_usd is None


def test_cost_per_pass_none_when_no_passes():
    [cell] = summarize([run("m", "low", "t", passed=False)], {"m": {"total_cost_usd": 0.5}})
    assert cell.cost_per_pass is None and cell.baseline_cost_usd == 0.5


def test_cells_sorted_by_task_model_effort():
    runs = [run("b", "xhigh", "t2"), run("a", "low", "t1"), run("a", "high", "t1"), run("a", "xhigh", "t1")]
    order = [(c.task, c.model, c.effort) for c in summarize(runs, {})]
    assert order == [("t1", "a", "low"), ("t1", "a", "high"), ("t1", "a", "xhigh"), ("t2", "b", "xhigh")]


def test_write_csv_and_markdown(tmp_path):
    cells = summarize([run("m", "low", "t"), run("m", "high", "t", cost=2.0)], {"m": {"total_cost_usd": 0.1}})
    write_csv(cells, tmp_path / "s.csv")
    rows = list(csv.DictReader((tmp_path / "s.csv").open()))
    assert len(rows) == 2 and rows[0]["cost_per_pass"] == "1.0"
    env = {"study_id": "s", "claude_code_version": "2.1.258", "recorded_at": "2026-09-03",
           "matrix": {"repeats": 3, "shuffle_seed": 7}, "task_fingerprints": {"t": "abcdef1234567890"}}
    write_markdown(cells, [env], {"m": {"total_cost_usd": 0.1}}, tmp_path / "s.md")
    text = (tmp_path / "s.md").read_text()
    assert "## Task: t" in text and "2.1.258" in text and "abcdef123456" in text
    assert "| m | low |" in text and "cost per pass" in text.lower()


def test_write_chart_is_optional(tmp_path):
    cells = summarize([run("m", "low", "t")], {})
    ok = write_chart(cells, tmp_path / "c.png")
    assert ok == (tmp_path / "c.png").exists()


def make_study(root, study_id, fp, model="m"):
    sd = root / study_id
    (sd / "runs" / f"{model}__low__t__r1").mkdir(parents=True)
    (sd / "baseline").mkdir()
    run(model, "low", "t", study_id=study_id).save(sd / "runs" / f"{model}__low__t__r1" / "meta.json")
    (sd / "environment.json").write_text(json.dumps({
        "study_id": study_id, "claude_code_version": "2.1.258", "recorded_at": "2026-09-03",
        "matrix": {"repeats": 1, "shuffle_seed": 1}, "task_fingerprints": {"t": fp}}))
    (sd / "baseline" / f"{model}.json").write_text(json.dumps({"model": model, "total_cost_usd": 0.2}))


def test_report_single_study(tmp_path):
    make_study(tmp_path, "s1", "fp1")
    out = report_study(tmp_path, ["s1"], log=lambda s: None)
    assert out == tmp_path / "s1" / "summary.md"
    assert (tmp_path / "s1" / "summary.csv").exists() and (tmp_path / "s1" / "runs.csv").exists()


def test_report_merge_and_fingerprint_refusal(tmp_path):
    make_study(tmp_path, "s1", "fp1", model="m1")
    make_study(tmp_path, "s2", "fp1", model="m2")
    out = report_study(tmp_path, ["s1", "s2"], log=lambda s: None)
    assert out == tmp_path / "merged" / "s1+s2" / "summary.md"
    assert "| s1 | m1 | low |" in out.read_text()
    make_study(tmp_path, "s3", "fp-different", model="m3")
    with pytest.raises(ReportError, match="'t'"):
        report_study(tmp_path, ["s1", "s3"], log=lambda s: None)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_report.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.report'`.

- [ ] **Step 3: Write the implementation**

`bench/report.py`:
```python
"""Aggregate run records into per-cell summaries: CSV, markdown, optional chart."""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .schema import RunMeta

EFFORT_ORDER = {"low": 0, "medium": 1, "high": 2, "xhigh": 3, "max": 4}


class ReportError(ValueError):
    """Studies cannot be merged, or a study folder is incomplete."""


@dataclass
class CellSummary:
    study_id: str
    model: str
    effort: str
    task: str
    n_completed: int
    n_pass: int
    pass_rate: float | None
    n_other: int
    tokens_total_median: float | None
    tokens_total_min: int | None
    tokens_total_max: int | None
    input_median: float | None
    cache_creation_median: float | None
    cache_read_median: float | None
    output_median: float | None
    thinking_median: float | None
    cost_median: float | None
    cost_min: float | None
    cost_max: float | None
    cost_per_pass: float | None
    turns_median: float | None
    minutes_median: float | None
    baseline_cost_usd: float | None
    tests_modified: int


def load_runs(results_root: Path, study_id: str) -> list[RunMeta]:
    runs_root = Path(results_root) / study_id / "runs"
    if not runs_root.exists():
        return []
    return [RunMeta.load(rd / "meta.json") for rd in sorted(runs_root.iterdir())
            if (rd / "meta.json").exists()]


def load_baselines(results_root: Path, study_id: str) -> dict[str, dict]:
    folder = Path(results_root) / study_id / "baseline"
    if not folder.exists():
        return {}
    out = {}
    for p in sorted(folder.glob("*.json")):
        data = json.loads(p.read_text())
        out[data.get("model", p.stem)] = data
    return out


def load_environment(results_root: Path, study_id: str) -> dict:
    p = Path(results_root) / study_id / "environment.json"
    if not p.exists():
        raise ReportError(f"study '{study_id}' has no environment.json; run preflight first")
    return json.loads(p.read_text())


def _median(values) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def summarize(runs: list[RunMeta], baselines: dict[str, dict]) -> list[CellSummary]:
    groups: dict[tuple, list[RunMeta]] = {}
    for r in runs:
        groups.setdefault((r.study_id, r.model, r.effort, r.task), []).append(r)
    cells = []
    for key in sorted(groups, key=lambda k: (k[3], k[0], k[1], EFFORT_ORDER.get(k[2], 9))):
        study_id, model, effort, task = key
        rs = groups[key]
        done = [r for r in rs if r.status == "completed"]
        n_pass = sum(1 for r in done if r.passed)
        costs = [r.total_cost_usd for r in done if r.total_cost_usd is not None]
        totals = [r.tokens_total for r in done if r.tokens_total is not None]
        baseline = baselines.get(model) or {}
        cells.append(CellSummary(
            study_id=study_id, model=model, effort=effort, task=task,
            n_completed=len(done), n_pass=n_pass,
            pass_rate=(n_pass / len(done)) if done else None,
            n_other=len(rs) - len(done),
            tokens_total_median=_median(totals),
            tokens_total_min=min(totals) if totals else None,
            tokens_total_max=max(totals) if totals else None,
            input_median=_median(r.input_tokens for r in done),
            cache_creation_median=_median(r.cache_creation_input_tokens for r in done),
            cache_read_median=_median(r.cache_read_input_tokens for r in done),
            output_median=_median(r.output_tokens for r in done),
            thinking_median=_median(r.thinking_tokens for r in done),
            cost_median=_median(costs),
            cost_min=min(costs) if costs else None,
            cost_max=max(costs) if costs else None,
            cost_per_pass=(sum(costs) / n_pass) if n_pass else None,
            turns_median=_median(r.num_turns for r in done),
            minutes_median=_median(r.duration_ms / 60000 for r in done),
            baseline_cost_usd=baseline.get("total_cost_usd"),
            tests_modified=sum(1 for r in done if r.visible_tests_modified),
        ))
    return cells


def write_csv(cells: list[CellSummary], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[f.name for f in fields(CellSummary)])
        writer.writeheader()
        for c in cells:
            writer.writerow(asdict(c))


def write_runs_csv(runs: list[RunMeta], path: Path) -> None:
    names = [f.name for f in fields(RunMeta)] + ["tokens_total"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for r in runs:
            row = asdict(r)
            row["tokens_total"] = r.tokens_total
            writer.writerow(row)


def _num(v, digits=0) -> str:
    if v is None:
        return "-"
    return f"{v:,.{digits}f}"


def _money(v) -> str:
    return "-" if v is None else f"${v:.2f}"


def write_markdown(cells: list[CellSummary], envs: list[dict], baselines: dict[str, dict], path: Path) -> None:
    lines = [f"# Study: {', '.join(e['study_id'] for e in envs)}", "", "## Methods", ""]
    for e in envs:
        m = e.get("matrix", {})
        lines.append(f"- **{e['study_id']}**: Claude Code {e.get('claude_code_version')}, recorded {e.get('recorded_at')}, "
                     f"repeats per cell {m.get('repeats')}, shuffle seed {m.get('shuffle_seed')}")
        for task, fp in sorted(e.get("task_fingerprints", {}).items()):
            lines.append(f"  - task `{task}` fingerprint `{fp[:12]}`")
    lines.append("- Cost is Claude Code's list-price estimate (`costBasis: list`), not a bill.")
    if baselines:
        lines.append("- Baseline overhead of a trivial prompt, per model:")
        for model, b in sorted(baselines.items()):
            usage = b.get("usage") or {}
            total = sum(usage.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                     "cache_read_input_tokens", "output_tokens"))
            lines.append(f"  - `{model}`: {_money(b.get('total_cost_usd'))}, {total:,} tokens")
    multi = len(envs) > 1
    for task in sorted({c.task for c in cells}):
        lines += ["", f"## Task: {task}", ""]
        head = (["Study"] if multi else []) + ["Model", "Effort", "Done", "Pass", "Cost per pass",
                "Tokens median (min to max)", "Input", "Cache write", "Cache read", "Output",
                "Thinking", "Cost median", "Turns", "Minutes", "Tests edited", "Other"]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        for c in [c for c in cells if c.task == task]:
            row = ([c.study_id] if multi else []) + [
                c.model, c.effort, str(c.n_completed),
                f"{c.n_pass}/{c.n_completed}" if c.n_completed else "-",
                _money(c.cost_per_pass) if c.cost_per_pass is not None else "no passes",
                f"{_num(c.tokens_total_median)} ({_num(c.tokens_total_min)} to {_num(c.tokens_total_max)})",
                _num(c.input_median), _num(c.cache_creation_median), _num(c.cache_read_median),
                _num(c.output_median), _num(c.thinking_median), _money(c.cost_median),
                _num(c.turns_median), _num(c.minutes_median, 1), str(c.tests_modified), str(c.n_other)]
            lines.append("| " + " | ".join(row) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def write_chart(cells: list[CellSummary], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    tasks = sorted({c.task for c in cells})
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        series: dict[str, list[tuple[int, float | None]]] = {}
        for c in cells:
            if c.task == task:
                label = f"{c.study_id}:{c.model}" if len({x.study_id for x in cells}) > 1 else c.model
                series.setdefault(label, []).append((EFFORT_ORDER.get(c.effort, 9), c.cost_per_pass))
        for label, pts in series.items():
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] if p[1] is not None else float("nan") for p in pts],
                    marker="o", label=label)
        ax.set_xticks(list(EFFORT_ORDER.values()))
        ax.set_xticklabels(list(EFFORT_ORDER.keys()))
        ax.set_title(task)
        ax.set_ylabel("cost per passed task (USD, list price)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def report_study(results_root: Path, study_ids: list[str], log=print) -> Path:
    results_root = Path(results_root)
    envs = [load_environment(results_root, s) for s in study_ids]
    seen: dict[str, tuple[str, str]] = {}
    for env in envs:
        for task, fp in env.get("task_fingerprints", {}).items():
            if task in seen and seen[task][1] != fp:
                raise ReportError(f"task '{task}' differs between studies {seen[task][0]} and {env['study_id']}; "
                                  "results are not comparable")
            seen.setdefault(task, (env["study_id"], fp))
    runs: list[RunMeta] = []
    baselines: dict[str, dict] = {}
    for s in study_ids:
        runs += load_runs(results_root, s)
        baselines.update(load_baselines(results_root, s))
    cells = summarize(runs, baselines)
    out_dir = results_root / study_ids[0] if len(study_ids) == 1 else results_root / "merged" / "+".join(study_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(cells, out_dir / "summary.csv")
    write_runs_csv(runs, out_dir / "runs.csv")
    write_markdown(cells, envs, baselines, out_dir / "summary.md")
    if write_chart(cells, out_dir / "summary.png"):
        log(f"chart written to {out_dir / 'summary.png'}")
    else:
        log("chart skipped: matplotlib is not installed (pip install -e '.[chart]')")
    log(f"report written to {out_dir / 'summary.md'}")
    return out_dir / "summary.md"
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_report.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add bench/report.py tests/test_report.py
git commit -m "feat: per-cell report with CSV, markdown, optional chart, study merge

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Preflight (`bench/preflight.py`)

**Files:**
- Create: `bench/preflight.py`, `tests/test_preflight.py`

**Interfaces:**
- Consumes: `build_env, build_probe_command, claude_version, config_dir` (Task 5); `Matrix` (Task 3); `fingerprint, load_task, TaskError` (Task 4).
- Produces: `PreflightResult(failures, warnings)` with `.ok`, `check_version(expected, actual) -> str | None`, `check_config_dir(cfg) -> tuple[list[str], list[str]]`, `check_login(probe_result) -> str | None`, `check_tasks(matrix, tasks_dir, results_root) -> tuple[list[str], dict[str, str]]`, `check_python() -> list[str]`, `run_probe(model, cwd) -> dict`, `write_environment(matrix, results_root, cc_version, fingerprints)`, `ensure_baselines(matrix, results_root, probe, log)`, `preflight(matrix, results_root, tasks_dir, log=print, probe=run_probe, version=claude_version) -> PreflightResult`, `PROBE_MODEL = "claude-haiku-4-5"`.
- Writes `environment.json` and `baseline/<model>.json` in the shapes Task 8 reads.

- [ ] **Step 1: Write the failing tests**

`tests/test_preflight.py`:
```python
import json

from bench.matrix import load_matrix
from bench.preflight import (PROBE_MODEL, check_config_dir, check_login,
                             check_tasks, check_version, preflight, run_probe)
from tests.conftest import FIXTURES

TASKS = FIXTURES / "tasks"


def test_check_version():
    assert check_version("2.1.258", "2.1.258") is None
    assert "2.1.259" in check_version("2.1.258", "2.1.259")


def test_check_config_dir_clean(bench_config):
    assert check_config_dir(bench_config) == ([], [])


def test_check_config_dir_missing(tmp_path):
    failures, _ = check_config_dir(tmp_path / "nope")
    assert failures and "bench login" in failures[0]


def test_check_config_dir_dirty(bench_config):
    (bench_config / "CLAUDE.md").write_text("rules")
    (bench_config / "settings.json").write_text(json.dumps({"hooks": {}, "model": "x"}))
    failures, warnings = check_config_dir(bench_config)
    assert any("CLAUDE.md" in f for f in failures)
    assert any("hooks" in f for f in failures)
    assert any("model" in w for w in warnings)


def test_check_login():
    assert check_login({"is_error": False, "result": "OK"}) is None
    assert "Not logged in" in check_login({"is_error": True, "result": "Not logged in · Please run /login"})


def test_check_tasks_detects_fingerprint_drift(toy_matrix_file, tmp_path):
    matrix = load_matrix(toy_matrix_file)
    failures, fps = check_tasks(matrix, TASKS, tmp_path / "results")
    assert failures == [] and set(fps) == {"toy-task"}
    sd = tmp_path / "results" / "toy-study"
    sd.mkdir(parents=True)
    (sd / "environment.json").write_text(json.dumps({"task_fingerprints": {"toy-task": "stale"}}))
    failures, _ = check_tasks(matrix, TASKS, tmp_path / "results")
    assert failures and "toy-task" in failures[0]


def test_check_tasks_missing_task(toy_matrix_file, tmp_path):
    toy_matrix_file.write_text(toy_matrix_file.read_text().replace('tasks = ["toy-task"]', 'tasks = ["ghost"]'))
    failures, _ = check_tasks(load_matrix(toy_matrix_file), TASKS, tmp_path / "results")
    assert failures and "ghost" in failures[0]


def test_run_probe_with_fake(fake_claude, bench_config, tmp_path):
    result = run_probe(PROBE_MODEL, tmp_path)
    assert result["is_error"] is False and result["result"] == "OK"


def test_preflight_happy_path_writes_environment_and_baselines(fake_claude, bench_config, toy_matrix_file, tmp_path):
    matrix = load_matrix(toy_matrix_file)
    calls = []

    def probe(model, cwd):
        calls.append(model)
        return run_probe(model, cwd)

    res = preflight(matrix, tmp_path / "results", TASKS, log=lambda s: None, probe=probe)
    assert res.ok, res.failures
    assert calls == [PROBE_MODEL, "claude-fake-a", "claude-fake-b"]
    env = json.loads((tmp_path / "results" / "toy-study" / "environment.json").read_text())
    assert env["claude_code_version"] == "2.1.258" and "toy-task" in env["task_fingerprints"]
    assert env["matrix"]["repeats"] == 2
    base = json.loads((tmp_path / "results" / "toy-study" / "baseline" / "claude-fake-a.json").read_text())
    assert base["model"] == "claude-fake-a" and base["total_cost_usd"] == 0.01
    # second call reuses baselines
    calls.clear()
    preflight(matrix, tmp_path / "results", TASKS, log=lambda s: None, probe=probe)
    assert calls == [PROBE_MODEL]


def test_preflight_refuses_on_version(fake_claude, bench_config, toy_matrix_file, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_VERSION", "3.0.0")
    res = preflight(load_matrix(toy_matrix_file), tmp_path / "results", TASKS, log=lambda s: None)
    assert not res.ok and any("3.0.0" in f for f in res.failures)
    assert not (tmp_path / "results" / "toy-study" / "baseline").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_preflight.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.preflight'`.

- [ ] **Step 3: Write the implementation**

`bench/preflight.py`:
```python
"""Refuse to start a study unless the environment is exactly what the results will claim."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .claude_cli import build_env, build_probe_command, claude_version, config_dir
from .matrix import Matrix
from .tasks import TaskError, fingerprint, load_task

PROBE_MODEL = "claude-haiku-4-5"
FORBIDDEN_ENTRIES = ("CLAUDE.md", "commands", "skills", "agents", "hooks")
FORBIDDEN_SETTINGS = ("hooks", "mcpServers", "enabledPlugins", "outputStyle", "statusLine")
OVERRIDDEN_SETTINGS = ("model", "effortLevel", "modelSettings")


@dataclass
class PreflightResult:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def check_version(expected: str, actual: str) -> str | None:
    if expected == actual:
        return None
    return (f"Claude Code version is {actual} but the matrix pins {expected}. "
            f"Either install that version or start a new study with claude_code_version = \"{actual}\".")


def check_config_dir(cfg: Path) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    cfg = Path(cfg)
    if not cfg.is_dir():
        return [f"benchmark config folder {cfg} does not exist. Run `bench login` for the one-time setup."], []
    for name in FORBIDDEN_ENTRIES:
        if (cfg / name).exists():
            failures.append(f"benchmark config folder contains {name}; it must stay clean. Remove {cfg / name}.")
    plugins = cfg / "plugins"
    if plugins.exists() and any(p.is_file() for p in plugins.rglob("*")):
        warnings.append(f"{plugins} contains files; make sure no plugin is enabled.")
    settings = cfg / "settings.json"
    if settings.exists():
        try:
            data = json.loads(settings.read_text() or "{}")
        except json.JSONDecodeError:
            failures.append(f"{settings} is not valid JSON")
            data = {}
        for key in FORBIDDEN_SETTINGS:
            if key in data and data[key]:
                failures.append(f"{settings} sets '{key}'; the benchmark config must not. Remove it.")
        for key in OVERRIDDEN_SETTINGS:
            if key in data:
                warnings.append(f"{settings} sets '{key}'; the runner overrides model and effort on the command line, so this is ignored.")
    return failures, warnings


def check_login(probe_result: dict) -> str | None:
    if probe_result.get("is_error"):
        return f"login check failed: {probe_result.get('result') or probe_result.get('error') or 'unknown error'}"
    return None


def check_tasks(matrix: Matrix, tasks_dir: Path, results_root: Path) -> tuple[list[str], dict[str, str]]:
    failures: list[str] = []
    fps: dict[str, str] = {}
    for name in matrix.tasks:
        try:
            fps[name] = fingerprint(load_task(name, tasks_dir))
        except TaskError as exc:
            failures.append(str(exc))
    env_path = Path(results_root) / matrix.study_id / "environment.json"
    if env_path.exists():
        recorded = json.loads(env_path.read_text()).get("task_fingerprints", {})
        for name, fp in fps.items():
            if name in recorded and recorded[name] != fp:
                failures.append(f"task '{name}' has changed since this study started (fingerprint "
                                f"{recorded[name][:12]} recorded, {fp[:12]} now). Start a new study id.")
    return failures, fps


def check_python() -> list[str]:
    failures = []
    if sys.version_info < (3, 11):
        failures.append(f"Python 3.11 or newer is required, found {platform.python_version()}")
    try:
        import pytest  # noqa: F401
    except ImportError:
        failures.append("pytest is not importable in this environment; run: pip install -e '.[dev]'")
    return failures


def run_probe(model: str, cwd: Path) -> dict:
    try:
        proc = subprocess.run(build_probe_command(model), cwd=cwd, env=build_env(),
                              capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"is_error": True, "result": "probe timed out after 300 seconds"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"is_error": True, "result": (proc.stderr or proc.stdout or "no output").strip()[:500]}


def write_environment(matrix: Matrix, results_root: Path, cc_version: str, fingerprints: dict[str, str]) -> None:
    sd = Path(results_root) / matrix.study_id
    sd.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_id": matrix.study_id,
        "claude_code_version": cc_version,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "config_dir": str(config_dir()),
        "matrix": asdict(matrix),
        "task_fingerprints": fingerprints,
    }
    (sd / "environment.json").write_text(json.dumps(payload, indent=2) + "\n")


def ensure_baselines(matrix: Matrix, results_root: Path, probe, log) -> list[str]:
    failures = []
    folder = Path(results_root) / matrix.study_id / "baseline"
    folder.mkdir(parents=True, exist_ok=True)
    for model in matrix.models:
        target = folder / f"{model}.json"
        if target.exists():
            continue
        log(f"recording baseline overhead for {model} ...")
        with tempfile.TemporaryDirectory(prefix="ccbench-baseline-") as tmp:
            result = probe(model, Path(tmp))
        if result.get("is_error"):
            failures.append(f"baseline probe for {model} failed: {result.get('result')}")
            continue
        target.write_text(json.dumps({
            "model": model,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_cost_usd": result.get("total_cost_usd"),
            "usage": result.get("usage"),
            "modelUsage": result.get("modelUsage"),
        }, indent=2) + "\n")
    return failures


def preflight(matrix: Matrix, results_root: Path, tasks_dir: Path, log=print,
              probe=run_probe, version=claude_version) -> PreflightResult:
    res = PreflightResult()
    res.failures += check_python()
    try:
        actual = version()
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        res.failures.append(f"could not run claude --version: {exc}")
        actual = None
    if actual is not None:
        problem = check_version(matrix.claude_code_version, actual)
        if problem:
            res.failures.append(problem)
    cfg_failures, cfg_warnings = check_config_dir(config_dir())
    res.failures += cfg_failures
    res.warnings += cfg_warnings
    task_failures, fps = check_tasks(matrix, tasks_dir, results_root)
    res.failures += task_failures
    if res.failures:
        _report(res, log)
        return res
    with tempfile.TemporaryDirectory(prefix="ccbench-login-") as tmp:
        problem = check_login(probe(PROBE_MODEL, Path(tmp)))
    if problem:
        res.failures.append(problem + " Run `bench login` for the one-time setup.")
        _report(res, log)
        return res
    write_environment(matrix, results_root, actual, fps)
    res.failures += ensure_baselines(matrix, results_root, probe, log)
    _report(res, log)
    return res


def _report(res: PreflightResult, log) -> None:
    for w in res.warnings:
        log(f"warning: {w}")
    for f in res.failures:
        log(f"FAIL: {f}")
    log("preflight OK" if res.ok else f"preflight failed with {len(res.failures)} problem(s)")
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_preflight.py -q`
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add bench/preflight.py tests/test_preflight.py
git commit -m "feat: preflight checks, environment record, baseline overhead probes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: CLI, study files, README (`bench/cli.py`)

**Files:**
- Create: `bench/cli.py`, `tests/test_cli.py`, `studies/smoke-opus5.toml`, `studies/2026-09-fable5-vs-fable51.toml`, `README.md`

**Interfaces:**
- Consumes: `load_matrix, MatrixError` (Task 3), `claude_version, config_dir` (Task 5), `preflight` (Task 9), `run_study` (Task 7), `score_study` (Task 6), `report_study, ReportError` (Task 8), `DEFAULT_TASKS_DIR` (Task 4).
- Produces: `main(argv=None) -> int` (console script `bench`), subcommands `login`, `preflight MATRIX`, `run MATRIX [--parallel N] [--only RUN_ID]`, `score MATRIX`, `report STUDY_ID [STUDY_ID ...]`, `all MATRIX [--parallel N]`; global options `--results DIR` (default `./results`) and `--tasks-dir DIR`.
- Exit codes: 0 ok, 1 study stopped or check failed, 2 bad matrix file.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
from bench.cli import main
from tests.conftest import FIXTURES

TASKS = FIXTURES / "tasks"


def test_login_prints_instructions(bench_config, capsys):
    assert main(["login"]) == 0
    out = capsys.readouterr().out
    assert "/login" in out and str(bench_config) in out and "CLAUDE_CONFIG_DIR" in out


def test_bad_matrix_exit_2(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text('study_id = "x"\n')
    assert main(["preflight", str(bad)]) == 2
    assert "matrix error" in capsys.readouterr().err


def test_all_end_to_end_with_fake(fake_claude, bench_config, toy_matrix_file, tmp_path, capsys):
    results = tmp_path / "results"
    rc = main(["--results", str(results), "--tasks-dir", str(TASKS), "all", str(toy_matrix_file)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "preflight OK" in out and "report written" in out
    summary = (results / "toy-study" / "summary.md").read_text()
    assert "## Task: toy-task" in summary and "| claude-fake-a | low |" in summary
    assert (results / "toy-study" / "runs.csv").exists()


def test_run_reports_stop_reason(fake_claude, bench_config, toy_matrix_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", "rate_limited")
    rc = main(["--results", str(tmp_path / "results"), "--tasks-dir", str(TASKS),
               "run", str(toy_matrix_file), "--parallel", "1"])
    assert rc == 1
    assert "STOPPED: rate limited" in capsys.readouterr().out


def test_report_missing_study_exit_1(tmp_path, capsys):
    assert main(["--results", str(tmp_path), "report", "ghost"]) == 1
    assert "environment.json" in capsys.readouterr().err
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.cli'`.

- [ ] **Step 3: Write the implementation**

`bench/cli.py`:
```python
"""Command line entry point: bench login | preflight | run | score | report | all."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .claude_cli import claude_version, config_dir
from .matrix import Matrix, MatrixError, load_matrix
from .preflight import preflight
from .report import ReportError, report_study
from .runner import run_study
from .scorer import score_study
from .tasks import DEFAULT_TASKS_DIR

LOGIN_TEXT = """One-time setup of the clean benchmark config folder.

Run these two commands in your terminal:

  mkdir -p {cfg}
  CLAUDE_CONFIG_DIR={cfg} claude

Inside Claude Code type  /login  and finish the browser prompt, then type  /exit

That folder is now logged in but has no CLAUDE.md, memory, plugins, hooks or
MCP servers. Preflight checks that it stays that way. Never put anything else in it.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description="Claude Code token-efficiency benchmark")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--results", type=Path, default=Path("results"),
                        help="results folder (default: ./results)")
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR,
                        help="folder holding the tasks (default: the repo's tasks/)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("login", help="print the one-time login instructions")
    for name, help_text in (("preflight", "check the environment, record it, run baseline probes"),
                            ("run", "run every unfinished run in the matrix (resumable)"),
                            ("score", "score completed runs that have no pass/fail yet"),
                            ("all", "preflight, run, score, report")):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("matrix", type=Path, help="study matrix .toml")
        if name in ("run", "all"):
            sp.add_argument("--parallel", type=int, default=None, help="runs at a time (default: matrix value)")
        if name == "run":
            sp.add_argument("--only", default=None, help="rerun exactly one run id")
    rp = sub.add_parser("report", help="aggregate one study, or merge several")
    rp.add_argument("study_ids", nargs="+")
    return parser


def _load(path: Path) -> Matrix | None:
    try:
        return load_matrix(path)
    except (MatrixError, OSError) as exc:
        print(f"matrix error: {exc}", file=sys.stderr)
        return None


def cmd_login(args) -> int:
    print(LOGIN_TEXT.format(cfg=config_dir()))
    return 0


def cmd_preflight(args) -> int:
    matrix = _load(args.matrix)
    if matrix is None:
        return 2
    return 0 if preflight(matrix, args.results, args.tasks_dir).ok else 1


def cmd_run(args) -> int:
    matrix = _load(args.matrix)
    if matrix is None:
        return 2
    if not preflight(matrix, args.results, args.tasks_dir).ok:
        return 1
    outcome = run_study(matrix, args.results, args.tasks_dir, claude_version(),
                        parallel=args.parallel, only=getattr(args, "only", None))
    print(f"executed {outcome.executed}, skipped {outcome.skipped}, "
          f"list-price spend so far ${outcome.spent_usd:.2f}")
    if outcome.stop_reason:
        print(f"STOPPED: {outcome.stop_reason}")
        print("Rerun the same command later to resume; finished runs are skipped.")
        return 1
    return 0


def cmd_score(args) -> int:
    matrix = _load(args.matrix)
    if matrix is None:
        return 2
    print(f"scored {score_study(matrix, args.results, args.tasks_dir)} run(s)")
    return 0


def cmd_report(args) -> int:
    try:
        report_study(args.results, args.study_ids)
    except ReportError as exc:
        print(f"report error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_all(args) -> int:
    matrix = _load(args.matrix)
    if matrix is None:
        return 2
    rc = cmd_run(args)
    if rc:
        return rc
    cmd_score(args)
    args.study_ids = [matrix.study_id]
    return cmd_report(args)


COMMANDS = {"login": cmd_login, "preflight": cmd_preflight, "run": cmd_run,
            "score": cmd_score, "report": cmd_report, "all": cmd_all}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_cli.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Write the two study files**

`studies/smoke-opus5.toml`:
```toml
# Pipeline smoke study: one run per task on Opus 5 at Claude Code's default effort.
# Both tasks must pass here before any Fable run is spent.
study_id = "smoke-opus5"
claude_code_version = "2.1.258"
models = ["claude-opus-5"]
efforts = ["xhigh"]
tasks = ["ledger-bugfix", "notes-feature"]
repeats = 1
parallel = 2
run_timeout_minutes = 45
budget_usd = 50
shuffle_seed = 1
```

`studies/2026-09-fable5-vs-fable51.toml`:
```toml
# First real study: Fable 5 vs Fable 5.1 across four effort levels.
# 2 models x 4 efforts x 2 tasks x 3 repeats = 48 runs.
study_id = "2026-09-fable5-vs-fable51"
claude_code_version = "2.1.258"
models = ["claude-fable-5", "claude-fable-5-1"]
efforts = ["low", "medium", "high", "xhigh"]
tasks = ["ledger-bugfix", "notes-feature"]
repeats = 3
parallel = 2
run_timeout_minutes = 45
budget_usd = 500
shuffle_seed = 20260902
```

Add to `tests/test_cli.py`:
```python
import pytest
from pathlib import Path
from bench.matrix import load_matrix

STUDIES = sorted((Path(__file__).resolve().parents[1] / "studies").glob("*.toml"))


@pytest.mark.parametrize("path", STUDIES, ids=[p.stem for p in STUDIES])
def test_shipped_studies_load(path):
    matrix = load_matrix(path)
    assert matrix.study_id == path.stem
```

Run: `python -m pytest tests/test_cli.py -q`
Expected: `7 passed`.

- [ ] **Step 6: Write README.md**

`README.md`:
```markdown
# cc-token-bench

Measures how many tokens, dollars, turns and minutes Claude Code spends to
finish a fixed coding task, across a grid of models and effort levels. Pass or
fail is decided by hidden tests. Every run starts from a fresh copy of the task
in a throwaway folder, driven through the real `claude` command line with a
clean config folder, so results are reproducible by anyone with a subscription.

Design spec: `docs/superpowers/specs/2026-09-02-cc-token-bench-design.md`.

## Reproduce a study in five commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,chart]"          # chart is optional
bench login                            # one time: prints two commands to run by hand
bench all studies/smoke-opus5.toml     # proves the pipeline and the tasks work
bench all studies/2026-09-fable5-vs-fable51.toml --parallel 2
```

If a run stops on a rate limit, rerun the same `bench all` command after the
printed reset time. Finished runs are skipped, never redone.

## What you get

`results/<study_id>/` holds `environment.json` (Claude Code version, task
fingerprints), one folder per run (`stream.jsonl`, `result.json`, `diff.patch`,
`junit.xml`, `meta.json`), `summary.csv`, `runs.csv`, `summary.md` (paste-ready
table) and `summary.png` when matplotlib is installed.

Headline metric per cell: **cost per passed task** = total list-price cost of
the cell's completed runs divided by its number of passes. Tokens are reported
in four buckets (input, cache write, cache read, output) plus thinking.

## Adding a model or a new study

Copy a file in `studies/`, change `study_id` and `models`, keep
`claude_code_version` equal to the version you are running, then
`bench all studies/<new>.toml`. Compare studies with
`bench report <old_id> <new_id>`; the report refuses if a task changed between
them (fingerprint check).

## Caveats to state when publishing

- Cost is Claude Code's list-price estimate (`costBasis: list`), not a bill.
- Three repeats per cell gives a range, not a confidence interval.
- The tasks are public, so future models may have seen them. Task fingerprints
  and versions are published for that reason.
- Results are tied to the pinned Claude Code version.

## Developing the harness

`python -m pytest -q` runs the harness tests against a fake `claude` executable
in `tests/fake_claude/`; no test spends money. `tests/test_task_integrity.py`
proves each benchmark task's visible tests pass on the template, its hidden
tests fail on the template, and both pass with the reference solution.
```

- [ ] **Step 7: Commit**

```bash
git add bench/cli.py tests/test_cli.py studies README.md
git commit -m "feat: bench CLI, study matrices, README

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Benchmark task A: `ledger-bugfix` + task integrity tests

**Files:**
- Create under `tasks/ledger-bugfix/`: `task.toml`, `PROMPT.md`, `template/README.md`, `template/ledger/__init__.py`, `template/ledger/models.py`, `template/ledger/parse.py`, `template/ledger/rates.py`, `template/ledger/reports.py`, `template/ledger/__main__.py`, `template/tests/test_ledger.py`, `hidden_tests/test_hidden_ledger.py`, `solution/ledger/rates.py`, `solution/ledger/reports.py`
- Create: `tests/test_task_integrity.py`

**Interfaces:**
- Consumes: `load_task, make_workdir, DEFAULT_TASKS_DIR` (Task 4), `run_hidden_tests` (Task 6).
- Produces: a task the runner can execute; `solution/` mirrors template paths for the files a correct fix changes (used only by the integrity test, never copied into a work folder).
- The three planted bugs: (1) `filter_by_range` excludes the end date although documented inclusive; (2) `convert` rounds every transaction to cents with banker's rounding, although the README says round once at the total with ROUND_HALF_UP; (3) `category_totals` accumulates into a mutable default dict across calls.

- [ ] **Step 1: Write the generic task integrity tests (they skip until a real task exists)**

`tests/test_task_integrity.py`:
```python
"""Every real task must: pass its visible tests on the template, fail its hidden
tests on the template, and pass both with the reference solution applied."""
import os
import shutil
import subprocess
import sys

import pytest

from bench.scorer import run_hidden_tests
from bench.tasks import DEFAULT_TASKS_DIR, load_task, make_workdir

REAL_TASKS = sorted(p.name for p in DEFAULT_TASKS_DIR.iterdir()
                    if (p / "task.toml").exists()) if DEFAULT_TASKS_DIR.exists() else []


def visible_tests_pass(workdir) -> bool:
    env = dict(os.environ, PYTHONPATH=str(workdir), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
                          cwd=workdir, env=env, capture_output=True, text=True)
    return proc.returncode == 0


def with_solution(task, tmp_path):
    workdir = make_workdir(task, base=tmp_path)
    shutil.copytree(task.root / "solution", workdir, dirs_exist_ok=True)
    return workdir


@pytest.mark.parametrize("name", REAL_TASKS)
def test_visible_tests_pass_on_template(name, tmp_path):
    assert visible_tests_pass(make_workdir(load_task(name), base=tmp_path))


@pytest.mark.parametrize("name", REAL_TASKS)
def test_hidden_tests_fail_on_template(name, tmp_path):
    task = load_task(name)
    total, passed = run_hidden_tests(make_workdir(task, base=tmp_path), task, tmp_path / "j.xml")
    assert total >= 3 and passed < total


@pytest.mark.parametrize("name", REAL_TASKS)
def test_everything_passes_with_solution(name, tmp_path):
    task = load_task(name)
    workdir = with_solution(task, tmp_path)
    assert visible_tests_pass(workdir)
    total, passed = run_hidden_tests(workdir, task, tmp_path / "j.xml")
    assert total >= 3 and passed == total


@pytest.mark.parametrize("name", REAL_TASKS)
def test_prompt_names_the_test_command_and_forbids_test_edits(name):
    prompt = load_task(name).prompt
    assert "python -m pytest tests" in prompt
    assert "Do not edit the existing test file" in prompt
```

Run: `python -m pytest tests/test_task_integrity.py -q`
Expected: `no tests ran` or all skipped (no real task yet).

- [ ] **Step 2: Write task.toml, PROMPT.md and the template README**

`tasks/ledger-bugfix/task.toml`:
```toml
name = "ledger-bugfix"
version = "1"
difficulty = "medium"
expected_minutes = 40
test_paths = ["tests/test_ledger.py"]
```

`tasks/ledger-bugfix/PROMPT.md`:
```
You are working in a small Python project called `ledger` (read README.md first).

Our finance volunteer reports three problems:

1. Monthly reports are missing every transaction dated on the last day of the month.
2. When she runs two category reports in the same Python session, the second report's totals include numbers from the first one.
3. Multi-currency totals are off by a cent or two compared with the bank statement. The rounding rule in README.md is the source of truth.

Find and fix the root causes in the library code under ledger/.

Rules:
- Keep the public function names in ledger/reports.py and ledger/rates.py.
- The existing tests must keep passing: `python -m pytest tests`
- Do not edit the existing test file. Put any new regression tests in a new file under tests/.
```

`tasks/ledger-bugfix/template/README.md`:
```markdown
# ledger

A small expense ledger. CSV in, reports out. Standard library only.

CSV columns: `date` (YYYY-MM-DD), `amount` (decimal), `currency` (3-letter code),
`category`, `note` (optional).

Command line: `python -m ledger FILE.csv --from 2026-08-01 --to 2026-08-31 --rate USD=1.63 --rate AUD=1.10`

## Money rules (source of truth)

- All money maths uses `decimal.Decimal`. Never floats.
- Convert each transaction to the base currency at full precision: `amount * rate`.
- Do not round individual transactions.
- Round once, at the final reported figure, to 2 decimal places using `ROUND_HALF_UP`.

## Date ranges

`filter_by_range(txs, start, end)` returns transactions dated from `start` to
`end`, inclusive at both ends. `month_range(year, month)` gives the first and
last day of a month for use with it.

## Reports

- `category_totals(txs, rates, base)` returns `{category: Decimal}` for the
  transactions given. A fresh call never includes data from an earlier call.
- `grand_total(txs, rates, base)` returns one `Decimal`.
- `monthly_summary(txs, rates, base)` returns `{"YYYY-MM": Decimal}`.

Tests: `python -m pytest tests`
```

- [ ] **Step 3: Write the template library**

`tasks/ledger-bugfix/template/ledger/__init__.py`:
```python
"""ledger: a small expense ledger. CSV in, reports out. See README.md."""
```

`tasks/ledger-bugfix/template/ledger/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal


@dataclass(frozen=True)
class Transaction:
    date: date
    amount: Decimal
    currency: str
    category: str
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal):
            raise TypeError("amount must be a Decimal")
        if len(self.currency) != 3:
            raise ValueError(f"currency must be a 3-letter code, got {self.currency!r}")
```

`tasks/ledger-bugfix/template/ledger/parse.py`:
```python
"""CSV parsing into Transaction objects."""
from __future__ import annotations

import csv
import io
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .models import Transaction

REQUIRED = ("date", "amount", "currency", "category")


class ParseError(ValueError):
    """The CSV is malformed. The message names the line."""


def parse_row(row: dict, line_no: int) -> Transaction:
    missing = [c for c in REQUIRED if not (row.get(c) or "").strip()]
    if missing:
        raise ParseError(f"line {line_no}: missing {', '.join(missing)}")
    try:
        when = date.fromisoformat(row["date"].strip())
    except ValueError as exc:
        raise ParseError(f"line {line_no}: bad date {row['date']!r}") from exc
    try:
        amount = Decimal(row["amount"].strip())
    except InvalidOperation as exc:
        raise ParseError(f"line {line_no}: bad amount {row['amount']!r}") from exc
    return Transaction(when, amount, row["currency"].strip().upper(),
                       row["category"].strip().lower(), (row.get("note") or "").strip())


def _parse_reader(reader: csv.DictReader) -> list[Transaction]:
    if reader.fieldnames is None:
        return []
    reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
    missing = [c for c in REQUIRED if c not in reader.fieldnames]
    if missing:
        raise ParseError(f"header is missing columns: {', '.join(missing)}")
    return [parse_row(row, i) for i, row in enumerate(reader, start=2)]


def parse_text(text: str) -> list[Transaction]:
    return _parse_reader(csv.DictReader(io.StringIO(text)))


def parse_csv(path: str | Path) -> list[Transaction]:
    with open(path, newline="", encoding="utf-8") as fh:
        return _parse_reader(csv.DictReader(fh))
```

`tasks/ledger-bugfix/template/ledger/rates.py`:
```python
"""Currency conversion."""
from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

CENTS = Decimal("0.01")


class RateError(KeyError):
    """No exchange rate is known for a currency."""


def rate_for(rates: dict[str, Decimal], currency: str, base: str) -> Decimal:
    """Units of `base` per one unit of `currency`."""
    if currency == base:
        return Decimal(1)
    try:
        return rates[currency]
    except KeyError:
        raise RateError(f"no rate for {currency} to {base}") from None


def convert(amount: Decimal, currency: str, rates: dict[str, Decimal], base: str) -> Decimal:
    """Convert one amount into the base currency."""
    return (amount * rate_for(rates, currency, base)).quantize(CENTS, rounding=ROUND_HALF_EVEN)
```

`tasks/ledger-bugfix/template/ledger/reports.py`:
```python
"""Filtering and totals."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from .models import Transaction
from .rates import CENTS, convert


def _round(value: Decimal) -> Decimal:
    return value.quantize(CENTS, rounding=ROUND_HALF_UP)


def month_range(year: int, month: int) -> tuple[date, date]:
    """First and last day of a month."""
    start = date(year, month, 1)
    following = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return start, following - timedelta(days=1)


def filter_by_range(txs, start: date, end: date) -> list[Transaction]:
    """Transactions dated from start to end, inclusive at both ends."""
    return [t for t in txs if start <= t.date < end]


def filter_by_category(txs, category: str) -> list[Transaction]:
    return [t for t in txs if t.category == category.strip().lower()]


def category_totals(txs, rates, base: str = "NZD", totals={}) -> dict[str, Decimal]:
    """Spend per category in the base currency, sorted by category name."""
    for t in txs:
        totals[t.category] = totals.get(t.category, Decimal(0)) + convert(t.amount, t.currency, rates, base)
    return {c: _round(v) for c, v in sorted(totals.items())}


def grand_total(txs, rates, base: str = "NZD") -> Decimal:
    return _round(sum((convert(t.amount, t.currency, rates, base) for t in txs), Decimal(0)))


def monthly_summary(txs, rates, base: str = "NZD") -> dict[str, Decimal]:
    """Spend per calendar month, keyed 'YYYY-MM'."""
    buckets: dict[str, Decimal] = defaultdict(Decimal)
    for t in txs:
        buckets[t.date.strftime("%Y-%m")] += convert(t.amount, t.currency, rates, base)
    return {k: _round(v) for k, v in sorted(buckets.items())}
```

`tasks/ledger-bugfix/template/ledger/__main__.py`:
```python
"""Command line: python -m ledger FILE.csv --from 2026-08-01 --to 2026-08-31 --rate USD=1.63"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from decimal import Decimal

from .parse import ParseError, parse_csv
from .reports import category_totals, filter_by_range, grand_total


def parse_rates(items: list[str]) -> dict[str, Decimal]:
    rates: dict[str, Decimal] = {}
    for item in items:
        code, _, value = item.partition("=")
        rates[code.strip().upper()] = Decimal(value.strip())
    return rates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ledger", description="Expense report for a date range.")
    parser.add_argument("csv")
    parser.add_argument("--from", dest="start", type=date.fromisoformat, required=True)
    parser.add_argument("--to", dest="end", type=date.fromisoformat, required=True)
    parser.add_argument("--base", default="NZD")
    parser.add_argument("--rate", action="append", default=[], metavar="CUR=RATE",
                        help="base units per 1 CUR, e.g. USD=1.63")
    args = parser.parse_args(argv)
    try:
        txs = filter_by_range(parse_csv(args.csv), args.start, args.end)
    except (ParseError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    rates = parse_rates(args.rate)
    for category, total in category_totals(txs, rates, args.base).items():
        print(f"{category:<16}{total:>12}")
    print(f"{'TOTAL':<16}{grand_total(txs, rates, args.base):>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write the visible tests (they pass on the buggy template)**

`tasks/ledger-bugfix/template/tests/test_ledger.py`:
```python
from datetime import date
from decimal import Decimal

import pytest

from ledger.__main__ import main
from ledger.models import Transaction
from ledger.parse import ParseError, parse_text
from ledger.rates import RateError, convert
from ledger.reports import (category_totals, filter_by_category, filter_by_range,
                            grand_total, month_range, monthly_summary)

RATES = {"USD": Decimal("1.60"), "AUD": Decimal("1.10")}


def tx(day, amount, currency="NZD", category="food"):
    return Transaction(date.fromisoformat(day), Decimal(amount), currency, category)


def test_parse_text_basic():
    txs = parse_text("date,amount,currency,category,note\n2026-08-01,12.50,nzd,Food,lunch\n")
    assert txs == [Transaction(date(2026, 8, 1), Decimal("12.50"), "NZD", "food", "lunch")]


def test_parse_missing_column():
    with pytest.raises(ParseError, match="missing columns"):
        parse_text("date,amount\n2026-08-01,1\n")


def test_parse_bad_amount_names_line():
    with pytest.raises(ParseError, match="line 3"):
        parse_text("date,amount,currency,category\n2026-08-01,1,NZD,a\n2026-08-02,x,NZD,a\n")


def test_convert_same_currency():
    assert convert(Decimal("5.00"), "NZD", RATES, "NZD") == Decimal("5.00")


def test_convert_usd():
    assert convert(Decimal("10.00"), "USD", RATES, "NZD") == Decimal("16.00")


def test_convert_unknown_currency():
    with pytest.raises(RateError):
        convert(Decimal("1"), "JPY", RATES, "NZD")


def test_month_range():
    assert month_range(2026, 2) == (date(2026, 2, 1), date(2026, 2, 28))
    assert month_range(2026, 12) == (date(2026, 12, 1), date(2026, 12, 31))


def test_filter_mid_range():
    txs = [tx("2026-08-01", "1"), tx("2026-08-15", "2"), tx("2026-09-02", "3")]
    assert filter_by_range(txs, date(2026, 8, 1), date(2026, 8, 31)) == txs[:2]


def test_filter_by_category():
    txs = [tx("2026-08-01", "1", category="food"), tx("2026-08-01", "2", category="fuel")]
    assert filter_by_category(txs, " Fuel ") == [txs[1]]


def test_category_totals_single_call():
    txs = [tx("2026-08-01", "10", category="food"), tx("2026-08-02", "5", category="fuel"),
           tx("2026-08-03", "2.50", category="food")]
    assert category_totals(txs, RATES) == {"food": Decimal("12.50"), "fuel": Decimal("5.00")}


def test_grand_total():
    assert grand_total([tx("2026-08-01", "10"), tx("2026-08-01", "10", "USD")], RATES) == Decimal("26.00")


def test_monthly_summary():
    txs = [tx("2026-07-31", "1"), tx("2026-08-01", "2")]
    assert monthly_summary(txs, RATES) == {"2026-07": Decimal("1.00"), "2026-08": Decimal("2.00")}


def test_cli_prints_totals(tmp_path, capsys):
    csv_file = tmp_path / "t.csv"
    csv_file.write_text("date,amount,currency,category\n2026-08-05,10,NZD,food\n2026-08-06,10,USD,fuel\n")
    assert main([str(csv_file), "--from", "2026-08-01", "--to", "2026-08-30", "--rate", "USD=1.60"]) == 0
    out = capsys.readouterr().out
    assert "food" in out and "fuel" in out and out.strip().endswith("26.00")
```

- [ ] **Step 5: Write the hidden tests**

`tasks/ledger-bugfix/hidden_tests/test_hidden_ledger.py`:
```python
from datetime import date
from decimal import Decimal

from ledger.models import Transaction
from ledger.reports import (category_totals, filter_by_range, grand_total,
                            month_range, monthly_summary)


def tx(day, amount, currency="NZD", category="food"):
    return Transaction(date.fromisoformat(day), Decimal(amount), currency, category)


def test_range_includes_end_date():
    txs = [tx("2026-08-30", "1"), tx("2026-08-31", "2"), tx("2026-09-01", "3")]
    assert filter_by_range(txs, date(2026, 8, 1), date(2026, 8, 31)) == txs[:2]


def test_month_report_includes_last_day():
    txs = [tx("2026-08-31", "4.00")]
    assert grand_total(filter_by_range(txs, *month_range(2026, 8)), {}) == Decimal("4.00")


def test_category_totals_fresh_between_calls():
    first = [tx("2026-08-01", "10")]
    second = [tx("2026-08-02", "5")]
    category_totals(first, {})
    assert category_totals(second, {}) == {"food": Decimal("5.00")}
    assert category_totals(second, {}) == {"food": Decimal("5.00")}


def test_no_per_transaction_rounding():
    rates = {"USD": Decimal("1.6349")}
    txs = [tx("2026-08-01", "10.01", "USD")] * 3
    assert grand_total(txs, rates) == Decimal("49.10")


def test_round_half_up_at_total():
    rates = {"USD": Decimal("2.3")}
    assert grand_total([tx("2026-08-01", "1.15", "USD")], rates) == Decimal("2.65")


def test_category_totals_round_half_up():
    rates = {"USD": Decimal("2.3")}
    assert category_totals([tx("2026-08-01", "1.15", "USD")], rates) == {"food": Decimal("2.65")}


def test_monthly_summary_no_per_transaction_rounding():
    rates = {"USD": Decimal("1.6349")}
    txs = [tx("2026-08-01", "10.01", "USD")] * 3
    assert monthly_summary(txs, rates) == {"2026-08": Decimal("49.10")}
```

- [ ] **Step 6: Write the reference solution**

`tasks/ledger-bugfix/solution/ledger/rates.py`:
```python
"""Currency conversion."""
from __future__ import annotations

from decimal import Decimal

CENTS = Decimal("0.01")


class RateError(KeyError):
    """No exchange rate is known for a currency."""


def rate_for(rates: dict[str, Decimal], currency: str, base: str) -> Decimal:
    """Units of `base` per one unit of `currency`."""
    if currency == base:
        return Decimal(1)
    try:
        return rates[currency]
    except KeyError:
        raise RateError(f"no rate for {currency} to {base}") from None


def convert(amount: Decimal, currency: str, rates: dict[str, Decimal], base: str) -> Decimal:
    """Convert one amount into the base currency at full precision (no rounding here)."""
    return amount * rate_for(rates, currency, base)
```

`tasks/ledger-bugfix/solution/ledger/reports.py`:
```python
"""Filtering and totals."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from .models import Transaction
from .rates import CENTS, convert


def _round(value: Decimal) -> Decimal:
    return value.quantize(CENTS, rounding=ROUND_HALF_UP)


def month_range(year: int, month: int) -> tuple[date, date]:
    """First and last day of a month."""
    start = date(year, month, 1)
    following = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return start, following - timedelta(days=1)


def filter_by_range(txs, start: date, end: date) -> list[Transaction]:
    """Transactions dated from start to end, inclusive at both ends."""
    return [t for t in txs if start <= t.date <= end]


def filter_by_category(txs, category: str) -> list[Transaction]:
    return [t for t in txs if t.category == category.strip().lower()]


def category_totals(txs, rates, base: str = "NZD") -> dict[str, Decimal]:
    """Spend per category in the base currency, sorted by category name."""
    totals: dict[str, Decimal] = {}
    for t in txs:
        totals[t.category] = totals.get(t.category, Decimal(0)) + convert(t.amount, t.currency, rates, base)
    return {c: _round(v) for c, v in sorted(totals.items())}


def grand_total(txs, rates, base: str = "NZD") -> Decimal:
    return _round(sum((convert(t.amount, t.currency, rates, base) for t in txs), Decimal(0)))


def monthly_summary(txs, rates, base: str = "NZD") -> dict[str, Decimal]:
    """Spend per calendar month, keyed 'YYYY-MM'."""
    buckets: dict[str, Decimal] = defaultdict(Decimal)
    for t in txs:
        buckets[t.date.strftime("%Y-%m")] += convert(t.amount, t.currency, rates, base)
    return {k: _round(v) for k, v in sorted(buckets.items())}
```

- [ ] **Step 7: Run the integrity tests**

Run: `python -m pytest tests/test_task_integrity.py -q`
Expected: `4 passed` (one task × four checks). If `test_hidden_tests_fail_on_template` fails, a planted bug is not being caught: re-read Step 3 against Step 5. If `test_everything_passes_with_solution` fails, compare Step 6 against Step 5 numerically before touching anything else.

- [ ] **Step 8: Commit**

```bash
git add tasks/ledger-bugfix tests/test_task_integrity.py
git commit -m "feat(tasks): ledger-bugfix task with hidden tests, reference solution, integrity tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Benchmark task B: `notes-feature`

**Files:**
- Create under `tasks/notes-feature/`: `task.toml`, `PROMPT.md`, `template/README.md`, `template/notes/__init__.py`, `template/notes/store.py`, `template/notes/cli.py`, `template/notes/__main__.py`, `template/tests/test_notes.py`, `hidden_tests/test_hidden_notes.py`, `solution/notes/store.py`, `solution/notes/cli.py`

**Interfaces:**
- Consumes: the integrity tests from Task 11 pick this task up automatically.
- Produces: a feature task whose hidden tests check exact command-line output. Both visible and hidden tests drive `python -m notes` in a subprocess with `NOTES_FILE` pointing at a temp file, so internal refactors by Claude cannot break the tests as long as the command line behaves.

- [ ] **Step 1: Write task.toml, PROMPT.md and the template README**

`tasks/notes-feature/task.toml`:
```toml
name = "notes-feature"
version = "1"
difficulty = "medium"
expected_minutes = 45
test_paths = ["tests/test_notes.py"]
```

`tasks/notes-feature/PROMPT.md`:
```
You are working in a small Python command-line tool called `notes` (read README.md first). Add tagging support. Exact behaviour required:

1. `notes add TITLE [--body BODY] [--tag TAG ...]`: `--tag` may be repeated. Tags are stored lowercase, without duplicates, sorted alphabetically. The output stays `added #ID`.
2. `notes list [--tag TAG]`: a note with tags prints as `{id:>3}  {title}  [{tags}]` where `{tags}` is the tags joined by a comma and a space, for example `  1  Plan  [alpha, work]`. A note with no tags keeps the current format. With `--tag`, only notes carrying that tag (compared case-insensitively) are listed; if none match, print `no notes`.
3. `notes tag ID TAG [TAG ...]` adds tags to an existing note with the same normalisation and prints `tagged #ID`. `notes untag ID TAG` removes one tag and prints `untagged #ID`; removing a tag the note does not have is not an error. An unknown ID prints a message on stderr and exits with code 1, as `show` does.
4. `notes show ID` prints a `tags: a, b` line immediately after the `created:` line when the note has tags, and no tags line otherwise.
5. `notes stats` prints exactly these lines:
   notes: N
   tagged: M
   untagged: K
   top tags:
     x (3)
   where N is the number of notes, M how many have at least one tag, K how many have none, and the `top tags:` line is followed by up to three lines, each indented by two spaces, showing the most-used tags as `tag (count)`, most used first, ties broken alphabetically. If no note has any tag, the last line is `top tags: none` and nothing follows it.
6. Notes files written before this change have no `tags` field. They must still load and count as having no tags.

Rules:
- Keep `python -m notes` as the entry point.
- The existing tests must keep passing: `python -m pytest tests`
- Do not edit the existing test file. Put tests for the new behaviour in a new file under tests/.
```

`tasks/notes-feature/template/README.md`:
```markdown
# notes

A tiny command-line notes tool. Standard library only. Notes live in one JSON
file, a list of objects with `id`, `title`, `body`, `created`. The file path
comes from the `NOTES_FILE` environment variable (default `notes.json` in the
current folder).

## Commands and exact output

| Command | Output |
|---|---|
| `python -m notes add TITLE [--body BODY]` | `added #ID` |
| `python -m notes list` | one line per note: `{id:>3}  {title}` (id right-aligned in 3 columns, two spaces, title); `no notes` when empty |
| `python -m notes show ID` | `#ID TITLE`, then `created: YYYY-MM-DD HH:MM`, then a blank line and the body if there is one |
| `python -m notes delete ID` | `deleted #ID` |

Unknown ID: a message on stderr, exit code 1.

Tests: `python -m pytest tests`
```

- [ ] **Step 2: Write the template package**

`tasks/notes-feature/template/notes/__init__.py`:
```python
"""notes: a tiny command-line notes tool. See README.md."""
```

`tasks/notes-feature/template/notes/store.py`:
```python
"""JSON-file storage. The file path comes from NOTES_FILE (default notes.json)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_FILE = "notes.json"


def notes_path() -> Path:
    return Path(os.environ.get("NOTES_FILE", DEFAULT_FILE))


@dataclass
class Note:
    id: int
    title: str
    body: str
    created: str

    @classmethod
    def from_dict(cls, data: dict) -> "Note":
        return cls(id=int(data["id"]), title=data["title"], body=data.get("body", ""),
                   created=data["created"])


def load(path: Path | None = None) -> list[Note]:
    p = path or notes_path()
    if not p.exists():
        return []
    raw = p.read_text(encoding="utf-8").strip()
    return [Note.from_dict(d) for d in json.loads(raw or "[]")]


def save(notes: list[Note], path: Path | None = None) -> None:
    p = path or notes_path()
    p.write_text(json.dumps([asdict(n) for n in notes], indent=2) + "\n", encoding="utf-8")


def next_id(notes: list[Note]) -> int:
    return max((n.id for n in notes), default=0) + 1


def add_note(title: str, body: str = "", path: Path | None = None) -> Note:
    notes = load(path)
    note = Note(next_id(notes), title.strip(), body.strip(),
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    notes.append(note)
    save(notes, path)
    return note


def get_note(note_id: int, path: Path | None = None) -> Note:
    for note in load(path):
        if note.id == note_id:
            return note
    raise KeyError(f"no note with id {note_id}")


def delete_note(note_id: int, path: Path | None = None) -> None:
    notes = load(path)
    kept = [n for n in notes if n.id != note_id]
    if len(kept) == len(notes):
        raise KeyError(f"no note with id {note_id}")
    save(kept, path)
```

`tasks/notes-feature/template/notes/cli.py`:
```python
"""Command line for notes. Entry point: python -m notes"""
from __future__ import annotations

import argparse
import sys

from . import store


def format_line(note: store.Note) -> str:
    return f"{note.id:>3}  {note.title}"


def cmd_add(args) -> int:
    note = store.add_note(args.title, args.body)
    print(f"added #{note.id}")
    return 0


def cmd_list(args) -> int:
    notes = store.load()
    if not notes:
        print("no notes")
        return 0
    for note in notes:
        print(format_line(note))
    return 0


def cmd_show(args) -> int:
    try:
        note = store.get_note(args.id)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    print(f"#{note.id} {note.title}")
    print(f"created: {note.created}")
    if note.body:
        print()
        print(note.body)
    return 0


def cmd_delete(args) -> int:
    try:
        store.delete_note(args.id)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    print(f"deleted #{args.id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="notes", description="Tiny notes tool.")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="add a note")
    add.add_argument("title")
    add.add_argument("--body", default="")
    add.set_defaults(func=cmd_add)
    sub.add_parser("list", help="list notes").set_defaults(func=cmd_list)
    show = sub.add_parser("show", help="show one note")
    show.add_argument("id", type=int)
    show.set_defaults(func=cmd_show)
    delete = sub.add_parser("delete", help="delete one note")
    delete.add_argument("id", type=int)
    delete.set_defaults(func=cmd_delete)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
```

`tasks/notes-feature/template/notes/__main__.py`:
```python
import sys

from .cli import main

sys.exit(main())
```

- [ ] **Step 3: Write the visible tests**

`tasks/notes-feature/template/tests/test_notes.py`:
```python
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(tmp_path, *args):
    env = {**os.environ, "NOTES_FILE": str(tmp_path / "notes.json"), "PYTHONPATH": str(ROOT)}
    proc = subprocess.run([sys.executable, "-m", "notes", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def test_list_empty(tmp_path):
    assert run(tmp_path, "list") == (0, "no notes\n", "")


def test_add_and_list(tmp_path):
    assert run(tmp_path, "add", "Buy milk") == (0, "added #1\n", "")
    assert run(tmp_path, "add", "Call mum")[1] == "added #2\n"
    assert run(tmp_path, "list")[1] == "  1  Buy milk\n  2  Call mum\n"


def test_show_with_body(tmp_path):
    run(tmp_path, "add", "Buy milk", "--body", "the oat one")
    code, out, _ = run(tmp_path, "show", "1")
    lines = out.splitlines()
    assert code == 0
    assert lines[0] == "#1 Buy milk"
    assert lines[1].startswith("created: ")
    assert lines[2] == "" and lines[3] == "the oat one"


def test_show_missing(tmp_path):
    code, out, err = run(tmp_path, "show", "7")
    assert code == 1 and out == "" and "7" in err


def test_delete(tmp_path):
    run(tmp_path, "add", "Buy milk")
    assert run(tmp_path, "delete", "1") == (0, "deleted #1\n", "")
    assert run(tmp_path, "list")[1] == "no notes\n"


def test_ids_keep_counting_after_delete(tmp_path):
    run(tmp_path, "add", "A")
    run(tmp_path, "add", "B")
    run(tmp_path, "delete", "2")
    assert run(tmp_path, "add", "C")[1] == "added #2\n"
```

- [ ] **Step 4: Write the hidden tests**

`tasks/notes-feature/hidden_tests/test_hidden_notes.py`:
```python
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(tmp_path, *args):
    env = {**os.environ, "NOTES_FILE": str(tmp_path / "notes.json"), "PYTHONPATH": str(ROOT)}
    proc = subprocess.run([sys.executable, "-m", "notes", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def test_old_file_without_tags_loads(tmp_path):
    (tmp_path / "notes.json").write_text(json.dumps(
        [{"id": 1, "title": "Old", "body": "", "created": "2026-01-01 00:00"}]))
    assert run(tmp_path, "list") == (0, "  1  Old\n", "")
    assert run(tmp_path, "stats")[1] == "notes: 1\ntagged: 0\nuntagged: 1\ntop tags: none\n"


def test_add_with_tags_is_normalised(tmp_path):
    assert run(tmp_path, "add", "Plan", "--tag", "Work", "--tag", "work", "--tag", "Alpha") == (0, "added #1\n", "")
    assert run(tmp_path, "list")[1] == "  1  Plan  [alpha, work]\n"


def test_untagged_note_keeps_old_format(tmp_path):
    run(tmp_path, "add", "Plain")
    run(tmp_path, "add", "Tagged", "--tag", "x")
    assert run(tmp_path, "list")[1] == "  1  Plain\n  2  Tagged  [x]\n"


def test_list_filter_by_tag_case_insensitive(tmp_path):
    run(tmp_path, "add", "A", "--tag", "x")
    run(tmp_path, "add", "B")
    assert run(tmp_path, "list", "--tag", "X")[1] == "  1  A  [x]\n"
    assert run(tmp_path, "list", "--tag", "zzz")[1] == "no notes\n"


def test_tag_and_untag(tmp_path):
    run(tmp_path, "add", "A")
    assert run(tmp_path, "tag", "1", "Foo", "bar") == (0, "tagged #1\n", "")
    assert run(tmp_path, "list")[1] == "  1  A  [bar, foo]\n"
    assert run(tmp_path, "untag", "1", "FOO") == (0, "untagged #1\n", "")
    assert run(tmp_path, "list")[1] == "  1  A  [bar]\n"
    assert run(tmp_path, "untag", "1", "nothere")[0] == 0


def test_tag_unknown_id_fails(tmp_path):
    code, out, err = run(tmp_path, "tag", "9", "x")
    assert code == 1 and out == "" and "9" in err


def test_show_tags_line(tmp_path):
    run(tmp_path, "add", "A", "--body", "b", "--tag", "t")
    lines = run(tmp_path, "show", "1")[1].splitlines()
    assert lines[0] == "#1 A" and lines[1].startswith("created: ") and lines[2] == "tags: t"
    assert lines[3] == "" and lines[4] == "b"
    run(tmp_path, "add", "B")
    assert not any(line.startswith("tags:") for line in run(tmp_path, "show", "2")[1].splitlines())


def test_stats(tmp_path):
    run(tmp_path, "add", "A", "--tag", "x", "--tag", "y")
    run(tmp_path, "add", "B", "--tag", "x")
    run(tmp_path, "add", "C")
    run(tmp_path, "add", "D", "--tag", "z", "--tag", "x")
    assert run(tmp_path, "stats")[1] == "notes: 4\ntagged: 3\nuntagged: 1\ntop tags:\n  x (3)\n  y (1)\n  z (1)\n"


def test_stats_top_three_only(tmp_path):
    for i, tag in enumerate("abcd"):
        run(tmp_path, "add", f"N{i}", "--tag", tag)
    run(tmp_path, "add", "E", "--tag", "d")
    assert run(tmp_path, "stats")[1] == "notes: 5\ntagged: 5\nuntagged: 0\ntop tags:\n  d (2)\n  a (1)\n  b (1)\n"
```

- [ ] **Step 5: Write the reference solution**

`tasks/notes-feature/solution/notes/store.py`:
```python
"""JSON-file storage. The file path comes from NOTES_FILE (default notes.json)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_FILE = "notes.json"


def notes_path() -> Path:
    return Path(os.environ.get("NOTES_FILE", DEFAULT_FILE))


def normalise_tags(tags: Iterable[str]) -> list[str]:
    return sorted({t.strip().lower() for t in tags if t.strip()})


@dataclass
class Note:
    id: int
    title: str
    body: str
    created: str
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Note":
        return cls(id=int(data["id"]), title=data["title"], body=data.get("body", ""),
                   created=data["created"], tags=normalise_tags(data.get("tags", [])))


def load(path: Path | None = None) -> list[Note]:
    p = path or notes_path()
    if not p.exists():
        return []
    raw = p.read_text(encoding="utf-8").strip()
    return [Note.from_dict(d) for d in json.loads(raw or "[]")]


def save(notes: list[Note], path: Path | None = None) -> None:
    p = path or notes_path()
    p.write_text(json.dumps([asdict(n) for n in notes], indent=2) + "\n", encoding="utf-8")


def next_id(notes: list[Note]) -> int:
    return max((n.id for n in notes), default=0) + 1


def add_note(title: str, body: str = "", tags: Iterable[str] = (), path: Path | None = None) -> Note:
    notes = load(path)
    note = Note(next_id(notes), title.strip(), body.strip(),
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), normalise_tags(tags))
    notes.append(note)
    save(notes, path)
    return note


def get_note(note_id: int, path: Path | None = None) -> Note:
    for note in load(path):
        if note.id == note_id:
            return note
    raise KeyError(f"no note with id {note_id}")


def _update(note_id: int, change: Callable[[Note], None], path: Path | None) -> Note:
    notes = load(path)
    for note in notes:
        if note.id == note_id:
            change(note)
            save(notes, path)
            return note
    raise KeyError(f"no note with id {note_id}")


def add_tags(note_id: int, tags: Iterable[str], path: Path | None = None) -> Note:
    def change(note: Note) -> None:
        note.tags = normalise_tags([*note.tags, *tags])
    return _update(note_id, change, path)


def remove_tag(note_id: int, tag: str, path: Path | None = None) -> Note:
    def change(note: Note) -> None:
        note.tags = [t for t in note.tags if t != tag.strip().lower()]
    return _update(note_id, change, path)


def delete_note(note_id: int, path: Path | None = None) -> None:
    notes = load(path)
    kept = [n for n in notes if n.id != note_id]
    if len(kept) == len(notes):
        raise KeyError(f"no note with id {note_id}")
    save(kept, path)
```

`tasks/notes-feature/solution/notes/cli.py`:
```python
"""Command line for notes. Entry point: python -m notes"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from . import store


def format_line(note: store.Note) -> str:
    line = f"{note.id:>3}  {note.title}"
    if note.tags:
        line += f"  [{', '.join(note.tags)}]"
    return line


def cmd_add(args) -> int:
    note = store.add_note(args.title, args.body, args.tag)
    print(f"added #{note.id}")
    return 0


def cmd_list(args) -> int:
    notes = store.load()
    if args.tag:
        wanted = args.tag.strip().lower()
        notes = [n for n in notes if wanted in n.tags]
    if not notes:
        print("no notes")
        return 0
    for note in notes:
        print(format_line(note))
    return 0


def cmd_show(args) -> int:
    try:
        note = store.get_note(args.id)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    print(f"#{note.id} {note.title}")
    print(f"created: {note.created}")
    if note.tags:
        print(f"tags: {', '.join(note.tags)}")
    if note.body:
        print()
        print(note.body)
    return 0


def cmd_delete(args) -> int:
    try:
        store.delete_note(args.id)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    print(f"deleted #{args.id}")
    return 0


def cmd_tag(args) -> int:
    try:
        store.add_tags(args.id, args.tags)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    print(f"tagged #{args.id}")
    return 0


def cmd_untag(args) -> int:
    try:
        store.remove_tag(args.id, args.tag)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    print(f"untagged #{args.id}")
    return 0


def cmd_stats(args) -> int:
    notes = store.load()
    tagged = sum(1 for n in notes if n.tags)
    print(f"notes: {len(notes)}")
    print(f"tagged: {tagged}")
    print(f"untagged: {len(notes) - tagged}")
    counts = Counter(tag for n in notes for tag in n.tags)
    if not counts:
        print("top tags: none")
        return 0
    print("top tags:")
    for tag, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]:
        print(f"  {tag} ({count})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="notes", description="Tiny notes tool.")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="add a note")
    add.add_argument("title")
    add.add_argument("--body", default="")
    add.add_argument("--tag", action="append", default=[])
    add.set_defaults(func=cmd_add)
    lst = sub.add_parser("list", help="list notes")
    lst.add_argument("--tag", default=None)
    lst.set_defaults(func=cmd_list)
    show = sub.add_parser("show", help="show one note")
    show.add_argument("id", type=int)
    show.set_defaults(func=cmd_show)
    delete = sub.add_parser("delete", help="delete one note")
    delete.add_argument("id", type=int)
    delete.set_defaults(func=cmd_delete)
    tag = sub.add_parser("tag", help="add tags to a note")
    tag.add_argument("id", type=int)
    tag.add_argument("tags", nargs="+")
    tag.set_defaults(func=cmd_tag)
    untag = sub.add_parser("untag", help="remove one tag from a note")
    untag.add_argument("id", type=int)
    untag.add_argument("tag")
    untag.set_defaults(func=cmd_untag)
    sub.add_parser("stats", help="tag statistics").set_defaults(func=cmd_stats)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
```

- [ ] **Step 6: Run the integrity tests for both tasks**

Run: `python -m pytest tests/test_task_integrity.py -q`
Expected: `8 passed` (two tasks × four checks). On the template, the hidden tests fail because `stats`, `tag`, `untag` and `--tag` are unknown arguments (argparse exits 2), which is the intended failure.

- [ ] **Step 7: Run the whole suite and commit**

Run: `python -m pytest -q`
Expected: all green, no test touched the real `claude`.

```bash
git add tasks/notes-feature
git commit -m "feat(tasks): notes-feature task with hidden tests and reference solution

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Real smoke study on Opus 5 (human gate, spends money)

This task runs the real Claude Code. It is the only task that does. Do not run it from a subagent; Benny or the main session runs it and reads the result.

**Files:**
- Create (by running, not by hand): `results/smoke-opus5/**`

**Interfaces:**
- Consumes: everything. Produces: proof that the pipeline and both tasks work end to end, plus the first committed results folder.

- [ ] **Step 1: One-time login into the clean config folder (Benny, by hand)**

Run: `bench login` and follow the two printed commands. Then confirm the folder is clean and logged in:

```bash
ls -la ~/.cc-bench
CLAUDE_CONFIG_DIR=~/.cc-bench claude -p "Reply with exactly the word OK and nothing else." --model claude-haiku-4-5 --output-format json | python3 -c "import json,sys; d=json.load(sys.stdin); print('logged in:', not d['is_error'], '| tokens:', sum(d['usage'][k] for k in ('input_tokens','cache_creation_input_tokens','cache_read_input_tokens','output_tokens')))"
```
Expected: `logged in: True | tokens: <well under 44,000>`. The token count is the clean baseline; write it down for the write-up.

- [ ] **Step 2: Preflight the smoke study**

Run: `bench preflight studies/smoke-opus5.toml`
Expected: `preflight OK`, plus `results/smoke-opus5/environment.json` and `results/smoke-opus5/baseline/claude-opus-5.json`. If it reports a version mismatch, edit `claude_code_version` in both study files to the installed version and rerun.

- [ ] **Step 3: Run the smoke study**

Run: `bench all studies/smoke-opus5.toml`
Expected: two lines like
```
claude-opus-5__xhigh__ledger-bugfix__r1   completed   tokens=...  cost=$...  turns=...  min=...  pass=True  util5h=...
claude-opus-5__xhigh__notes-feature__r1   completed   tokens=...  cost=$...  turns=...  min=...  pass=True  util5h=...
```
then `report written to results/smoke-opus5/summary.md`.

- [ ] **Step 4: Read the evidence, not just the summary**

For each run folder under `results/smoke-opus5/runs/`:
- `diff.patch`: does the change look like a real fix or feature, not a test edit? `meta.json` must show `"visible_tests_modified": false`.
- `junit.xml` and `meta.json`: `tests_passed == tests_total`.
- `stream.jsonl` first line: `"cwd"` is a `ccbench-...` temp folder, `"mcp_servers": []`, and `"slash_commands"` is a short default list (no personal skills).

If a task did not pass on Opus 5 at xhigh, the task is broken or too hard. Fix the task (prompt wording or hidden test), bump `version` in its `task.toml`, delete `results/smoke-opus5/`, and rerun Steps 2 and 3. Do not change the harness to make a task pass.

- [ ] **Step 5: Commit the smoke results**

```bash
git add results/smoke-opus5
git commit -m "results: smoke study on Opus 5, both tasks pass

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

- [ ] **Step 6: Hand-off note for the real study (not part of this plan's build)**

The Fable study is started with `bench all studies/2026-09-fable5-vs-fable51.toml --parallel 2`. Expect 4 to 8 hours. If it stops on a rate limit, rerun the same command after the printed reset time. When it finishes, commit `results/2026-09-fable5-vs-fable51/` and write the post from its `summary.md`.

---

## Plan self-review (done at authoring time)

- **Spec coverage:** 5.1 matrix → Task 3; 5.2 tasks + fingerprint → Tasks 4, 11, 12; 5.3 preflight + baselines + environment.json → Task 9; 5.4 runner (fresh copy, command line, timeout, classification, rate-limit stop, budget cap, resume, `--only`, per-run line) → Task 7; 5.5 scorer (hidden tests, cheat check, rebuild from patch, `bench score`) → Task 6; 5.6 report (cells, CSV, markdown, chart, merge with fingerprint refusal) → Task 8; 5.7 CLI incl. `login` → Task 10; 6 isolation → Tasks 5, 9; 7 error handling → Task 7 (statuses, traceback file), Task 9; 8 fake claude + scenario list → Task 5, tests in Tasks 6 to 10; 9 meta.json fields → Task 2 (`passed` instead of `pass`); 10 protocol → README (Task 10) + Task 13; 11 caveats → README.
- **Deviations from the spec, deliberate:** rate-limited runs are moved to `results/<study>/rate_limited/` instead of deleted, so the evidence survives while the run id is freed for retry. The `notes list` format keeps the old shape for untagged notes so the visible test stays valid and the cheat flag stays meaningful. Preflight's login check uses Haiku, not Opus 5, because it only proves login works.
- **Type consistency:** `RunMeta.passed` (Task 2) is what Tasks 6, 7, 8 read; `StudyOutcome(executed, skipped, spent_usd, stop_reason)` (Task 7) is what Task 10 prints; `run_hidden_tests(workdir, task, junit_path) -> (total, passed)` (Task 6) is used by Task 11's integrity tests; `environment.json` keys written in Task 9 (`task_fingerprints`, `claude_code_version`, `recorded_at`, `matrix`) are the keys Task 8 reads.
