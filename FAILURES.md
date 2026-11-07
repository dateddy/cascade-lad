# Failure log

Every agent mistake the user corrects gets an entry here, newest last, no ids (so two branches can both add entries and merge cleanly). An entry must name a **control**.
Prefer a computational control (script, hook, check that always runs) over a new line of instruction.

Entry format:

```
## <YYYY-MM-DD> · <short title>
- What happened: <observable facts>
- Root cause: <why it was possible>
- Control: <what now prevents it; file or script names>
```
