# Python Development Guidelines

## Core Principles
- **Keep it simple**: Write the minimal code necessary to solve the problem
- **Modular design**: Break functionality into small, focused modules and functions
- **Readability first**: Code should be self-documenting and easy to understand

## Code Style
- Follow PEP 8 conventions
- Use meaningful variable and function names
- Keep functions small and focused on a single responsibility
- Prefer composition over inheritance

## Best Practices
- Maintain consistency with existing code style and patterns; when in doubt, follow the conventions already established in the codebase
- Use modern type hints with built-in types (`list[str]`, `dict[str, int]`) instead of typing module imports (`List`, `Dict`); only import from typing when necessary (`Union`, `Optional`, `Protocol`, `TypeVar`); prefer `str | None` over `Optional[str]` and `int | str` over `Union[int, str]` in Python 3.10+
- Handle exceptions gracefully with specific exception types
- Write docstrings for public modules, classes, and functions (following PEP 257); for simple, straightforward functions, a concise one-line docstring is sufficient
- Use list/dict comprehensions when they improve readability
- Avoid deep nesting - prefer early returns and guard clauses

## Project Structure
- Keep the project structure flat and simple
- Group related functionality into modules
- Use `__init__.py` files to control module exports
- Separate concerns: data, logic, and presentation

## Dependencies
- Minimize external dependencies
- Use standard library when possible
- Pin dependency versions for reproducibility
- Document why each dependency is needed

## Git Guidelines
- Keep commit messages concise and clear
- Follow the 50/72 rule: subject line ≤50 chars, body lines ≤72 chars
- Use imperative mood in subject line (e.g., "Add feature" not "Added feature")
- No emojis or decorative elements
- No attribution or generation indicators
- Subject line only for simple changes
- Add body with context only when necessary to explain why the change was made
- Use lists in the body when elaborating on multiple changes

## Commands
Run linting and type checking:
```bash
# Linting and formatting with ruff
python -m ruff check .
python -m ruff format .
```