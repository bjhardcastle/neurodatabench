# AGENTS.md

## Project Overview
- Audience: Tool maintainers comparing PyNWB, LINDI/Neurosift, storage/cache backends, other NWB reader implementations, and developers who have their own NWB access solution and want to benchmark it fairly.

## Living Agent Docs
- Treat `AGENTS.md` and `CONTEXT.md` as living agent-facing project guidance and memory, respectively.
- Update `AGENTS.md` when you discover or are prompted to use durable ways of working that future agents should follow: commands, conventions, boundaries, benchmark design rules, validation expectations, or project-specific gotchas.
- Update `CONTEXT.md` when you make meaningful progress, clarify the current goal, change the project shape, discover blockers, or leave useful handoff notes.
- Prefer small, additive edits: append a bullet, tighten an outdated sentence, or replace a stale detail with the current truth.
- Do not wholesale delete guidance in `AGENTS.md` unless explicitly asked.
- When a code change affects documented behavior, update the relevant agent docs in the same change.

## Development Rules
- Non-trivial Python modules should have a module docstring that explains their role.
- All classes and functions must have docstrings.
- Use absolute imports, e.g. `import neurodatabench.runner` not `from neurodatabench import runner`. The exception is idiomatic stdlib imports, such as `from typing import Literal`

## Git Workflow
- Keep generated data-heavy artifacts (databases, caches, binary files) out of git.
- Include a `Co-authored-by:` trailer using the agent's own name and email.
