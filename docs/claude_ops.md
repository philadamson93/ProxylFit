# Claude Code Operating Standards

This document defines how Claude Code should operate in this repository. Reference this file at the start of every planning document.

---

## Core Principles

1. **Plan before you code.** Always enter Plan mode (Shift+Tab twice) before writing any code. Iterate on the plan until it's solid, then execute.

2. **Re-enter plan mode when direction changes.** If you discover a new issue, architectural concern, or change in direction while implementing, pause and re-enter plan mode to get feedback before continuing.

3. **A wrong fast answer is slower than a right slow answer.** Prioritize correctness over speed. Use thinking mode for complex tasks.

4. **You don't trust; you instrument.** Always provide verification mechanisms. Feedback loops multiply output quality 2-3x.

5. **YAGNI (You Aren't Gonna Need It).** Don't build for hypothetical futures. Implement what's needed now, nothing more.

---

## Planning Workflow

### Starting a Task

1. Enter Plan mode before any implementation
2. **Read relevant documentation first.** Search `docs/` and the codebase for existing patterns, utilities, and context before proposing solutions. Understand what exists before suggesting changes.
3. Draft the plan in plan mode's internal file (the only file plan mode allows writing to)
4. Begin the plan document with:
   ```
   Reference: docs/claude_ops.md
   ```
5. Articulate both *what* you're building and *why*
6. Ask: "Are there any points of ambiguity about this plan?" to surface underspecified requirements
7. Iterate on the plan until solid, then exit plan mode

### Saving the Plan (after exiting plan mode)

**Important: Plan mode limitation.** Claude Code's plan mode can only write to its internal plan file (`~/.claude/plans/`). It **cannot** write to `docs/plans/` in the repo. This creates a two-step process:

1. **Exit plan mode** — this approves the *plan content*, not implementation
2. **Immediately save to `docs/plans/`** — copy the plan to the repo with a descriptive filename (e.g., `add-temporal-loss-weighting.md`, not `plan_01.md`). This ensures traceability and allows the user to review plans across sessions.
3. **Stop and confirm** — ask the user before starting implementation. Do not create task lists, write code, or make any changes beyond saving the plan doc.

Exiting plan mode ≠ "start coding." Treat it as "plan content approved, now persist it."

### After Completing a Plan

- **Update all affected documentation** when a plan is implemented. Fix stale paths, CLI examples, import references, and cross-links in `docs/`.
- **Mark plan docs as completed** by adding `**Status: Completed** (date)` at the top.
- **Update the plans README** (`docs/plans/README.md`) feature table with the new status.

### When to Re-enter Plan Mode

- Discovering the current approach won't work
- Uncovering a new requirement or constraint
- Realizing the scope is larger than expected
- Finding an architectural issue that affects the design
- Any time you're uncertain whether to proceed

### Plan Document Structure

```markdown
Reference: docs/claude_ops.md

# [Descriptive Task Title]

## Goal
What are we building and why?

## Approach
How will we implement this?

## Files to Modify
- path/to/file.py - description of changes

## Open Questions
- Any ambiguities to resolve?

## Verification
How will we know this works?
```

---

## Code Quality Standards

### Re-use Over Duplication

- Always check for existing utilities before writing new code
- Extend existing classes/functions rather than creating parallel implementations
- Prioritize modularity and clean code over expediency

### Simplicity

- Write the simplest code that solves the problem
- Avoid unnecessary abstractions
- Don't add features that aren't explicitly requested

---

## Git Practices

### Branch Awareness

- **Always check and report the current branch** before making commits or suggesting git operations
- Confirm you're on the expected branch before proceeding with changes

### Feature Branching

- **Major changes should be made in a new feature branch**, not directly on main.
- Documentation updates and minor bug fixes can go directly on main.

### Commit Messages

- **No AI attribution.** Never include "Co-Authored-By: Claude" or similar
- **One sentence per commit.** Keep messages concise and descriptive
- **Thematic separation.** Split changes into separate commits by theme:
  - One commit for config changes
  - Another for core logic changes
  - Another for documentation updates

### Commit Frequency

- Commit frequently to maintain clean revert points
- Each commit should represent a coherent, working state

---

## Communication Standards

### Ask Clarifying Questions For:

- Functional requirements (what to build, how it should behave)
- Ambiguous specifications
- Decisions that significantly affect architecture
- Anything where assumptions could lead to wasted work
- **Fallback vs exception behavior**: Don't assume fallbacks are preferred — they can mask upstream errors. Ask the user explicitly.
- **Testing plans**: Brainstorm which aspects are testable, critical to test, and what can be mocked vs needs integration testing. Get user input before writing tests.

### Use Your Judgement For:

- Implementation details (variable names, code patterns)
- Internal structure decisions
- Standard refactoring choices
- Obvious bug fixes

### Document Non-Obvious Decisions

If you make a choice that isn't obvious, note it briefly in:
- Code comments (sparingly)
- Commit messages
- The planning document

---

## Institutional Memory

### When Claude Makes Mistakes

Add learnings to [`docs/lessons.md`](lessons.md) so they don't repeat. Examples:
- "Don't modify X without also updating Y"
- "Always run Z before committing changes to W"
- "The config parameter `foo` must be set when using feature `bar`"

### When Patterns Emerge

Document recurring patterns in the appropriate `docs/` file to help future sessions.

---

## Context Management

- **Fresh sessions for fresh tasks.** Start new sessions when switching to unrelated work
- **Match rigor to stakes.** Prototypes allow looser constraints; production changes require thorough planning and review

---

## Pre-Commit Review Agents

Before committing, run two independent review passes using subagents. These are mandatory for any change that touches production code or tests.

**This is a live development machine.** Agents can and should run code, launch the GUI, take screenshots, and verify features directly. Use `uv run` for all Python commands.

### Test Review Agent

**Only run this agent if new or substantially modified tests were written.** Moving imports or deleting dead test code does not require a test review.

After writing tests, spawn a subagent to review all new/modified test code. The agent should check:

- **Behavior over implementation.** Tests should assert observable outcomes (query results, DataFrame contents, return values), not implementation details (string counts, internal SQL structure, JOIN types).
- **Fragility.** Would the test break if an unrelated part of the code changes? If yes, rewrite.
- **Coverage.** Do the tests cover the cases listed in the plan doc? Any gaps?
- **False positives.** Could the test pass even if the feature is broken? (e.g., an assertion that's always true)
- **Fixture reuse.** Are existing fixtures reused where possible, or is there unnecessary duplication?
- **Data contracts.** Do tests validate critical stage-boundary contracts (required fields, nullability, key uniqueness, and type invariants)?

Prompt template:
```
IMPORTANT: Read docs/claude_ops.md first.

Review all new/modified test code in this session. For each test, assess:
1. Does it test behavior or implementation details?
2. Is it fragile (would unrelated changes break it)?
3. Does it match the plan doc spec?
4. Could it produce false positives?

Then assess test coverage gaps for the implemented features:
- Unit tests: Are there missing unit tests for new functions, edge cases, or error paths?

After reviewing, run the tests: uv run pytest tests/ -v
If GUI tests exist, run them and capture screenshots for visual verification.

Report issues with specific file:line references and suggested fixes.
```

### Implementation Review Agent

After completing implementation, spawn a separate subagent to review all changes for fidelity and standards. The agent should check:

- **Plan fidelity.** Do the code changes match what the plan doc specifies? Any deviations, missing pieces, or scope creep?
- **claude_ops compliance.** Were operating standards followed? (plan doc created, thematic commits, no AI attribution, docs updated, etc.)
- **Code quality.** YAGNI, simplicity, no unnecessary abstractions, re-use over duplication.
- **Completeness.** Are all files listed in the plan's "Files to Modify" section actually modified? Are docs/README updated?
- **Security.** No PHI exposure, no credentials, safe SQL construction.

Prompt template:
```
IMPORTANT: Read docs/claude_ops.md first.

Review all changes in this session against the plan doc at docs/plans/<plan>.md and the standards in docs/claude_ops.md. Check:
1. Do changes match the plan spec exactly?
2. Were claude_ops procedures followed?
3. Any code quality issues (YAGNI violations, unnecessary complexity)?
4. Are all docs updated?

After reviewing, verify live:
1. Run affected tests: uv run pytest tests/ -v
2. If UI was changed, launch the app and take a screenshot to verify visually
3. If export functionality was changed, test the export and verify the output file

Report issues with specific file:line references.
```

---

## Verification

Always verify changes by running code. Don't describe what to check — actually check it.

### Verification Levels (use the highest applicable)

1. **Unit tests**: `uv run pytest tests/ -v`
2. **Script verification**: Write and run a short Python script that exercises the change
3. **GUI verification**: Launch the app, take screenshots, read them with the `Read` tool
4. **Round-trip testing**: Export a file → verify it exists → re-import → verify app state

### Screenshot-Verify Loop

Claude can visually inspect the running application:

1. Launch the app or a script that opens a dialog
2. Capture: `screencapture -x /tmp/proxylfit_screenshot.png`
   - Or from within a test: `widget.grab().save("/tmp/screenshot.png")`
3. Read the screenshot with the `Read` tool (Claude sees the image visually)
4. Analyze: check layout, button states, plot content, error messages
5. Interact and repeat

### Peekaboo MCP (Full App Automation)

When Peekaboo is available, Claude can interact with the live running app directly:
- `see` — capture a screenshot of any window
- `click` — click at coordinates or on identified UI elements
- `type` — type text into the focused element
- `press` / `hotkey` — keyboard shortcuts

Use Peekaboo for end-to-end workflow testing: load data → navigate menus → export files → verify outputs.

### Available Tools

| Tool | Use For |
|------|---------|
| `uv run pytest` | Automated tests |
| `pytest-qt` / `qtbot` | Headless GUI test scripts with signal waiting |
| `QTest.mouseClick()` / `QTest.keyClicks()` | In-process widget interaction |
| `QWidget.grab()` | Widget-level screenshots (no permissions needed) |
| `screencapture -x file.png` | OS-level screenshots |
| Peekaboo `see` / `click` / `type` | Full app automation via MCP |
| `Read` tool on .png | Visual verification (Claude sees the image) |
