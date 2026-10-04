# Domain Documentation

- **Layout**: Single-context
- **Core Domain Context**: [CONTEXT.md](../../CONTEXT.md)
- **Architecture & Specifications**: [PROJECT.md](../../PROJECT.md)
- **ADR Directory**: `docs/adr/` (created as needed)

## Consumer Rules
- Consult `CONTEXT.md` for target class definitions (Fire = class 0, Smoke = class 1, Ambient Frame = $\max(P) < \tau$) and safety constraints ($\text{Fire Recall} \ge 90\%$).
- Adhere to `PROJECT.md` interface contracts for all `src/` modules.
