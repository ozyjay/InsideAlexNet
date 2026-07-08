# AGENTS.md

Guidance for future development of **How Does a Neural Network See?**

## Working rules

- Plan before changing code.
- Make small incremental changes.
- Keep fallback mode working.
- Avoid unnecessary dependencies.
- Keep the app local-first.
- Use FastAPI for the local app unless there is an explicit decision to change stacks.
- Do not add webcam, upload, or data storage without an explicit decision.
- Do not overclaim AI capabilities.
- Add or update tests with changes.
- Preserve reset behaviour.

## Public wording

Use accurate public wording such as:

> “This shows how a trained vision model responds at different layers.”

Do not describe the demo as showing private reasoning, human-like sight, or guaranteed truth.
