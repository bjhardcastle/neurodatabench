# AGENTS.md

## Project Overview
- Audience: Tool maintainers comparing PyNWB, LINDI/Neurosift, storage/cache backends, other NWB reader implementations, and developers who have their own NWB access solution and want to benchmark it fairly.

## Development Rules
- Non-trivial Python modules should have a module docstring that explains their role.
- All classes and functions must have docstrings.
- Internally, use absolute imports, e.g. `import neurodatabench.runner` not `from neurodatabench import runner`. The exception is idiomatic stdlib imports, such as `from typing import Literal`. User-facing templates should use relative imports for brevity.
- Use type hints everywhere
- Use debug logging everywhere except start/finish messages
- mark internal api with underscore prefixes.
- keep it simple to use for users but production-level.
- For supervised benchmark runs with an output directory, pass the same directory through `--timeout-profile-out` so memory and network samples survive a forced timeout.

## Git Workflow
- Keep generated data-heavy artifacts (databases, caches, binary files) out of git.
- Include a `Co-authored-by:` trailer using the agent's own name and email.
