# Python project starter

A minimal Python workspace with an entry point, tests directory, and documentation
directory. No application algorithms or runtime dependencies are included yet.

## Layout

```text
main.py          Application entry point
pyproject.toml   Project metadata and tool configuration
tests/           Future automated tests
docs/            Future documentation
```

## Setup

Use Python 3.11 or newer. Create and activate a local virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Install the development tools:

```sh
python -m pip install 'pytest>=8' 'ruff>=0.9'
```

These match the `dev` dependency group in `pyproject.toml`. Pytest runs automated
tests; Ruff checks code quality and formats Python files. Runtime libraries can
be added to `project.dependencies` when needed.

## Run

```sh
python main.py
```

The entry point currently does nothing and exits successfully.

## Development checks

```sh
ruff check .
ruff format --check .
python -m pytest
```

The tests directory starts empty. Until tests are added, pytest reports no tests
collected and exits with code 5.
