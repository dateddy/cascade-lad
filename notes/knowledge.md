# Knowledge

Facts about this repo and its environment that took effort to find out. The agent adds a line here when it needed
3 or more tries to find the right way, or when you corrected a wrong claim about the repo.
Line format: `- <fact>; <the right way> (verified YYYY-MM-DD: <command or evidence>)`.
Not here: bugs (dbg journal), decisions (decisions.md), agent mistakes (FAILURES.md).

- Two verify scripts read .agentkit.json differently: .agentkit/scripts/verify.py (hook, unified keys verify.tests) and the skill vf .agentkit/skills/verifying-changes/scripts/verify.py (reads verify.test only, else probes system python, which lacks coverage, and skips tests); keep both verify.test and verify.tests until the skill copy is upgraded, and check with `vf --detect` that the test line says "set in .agentkit.json" (verified 2026-11-19: vf --detect after dropping verify.test showed "pytest not runnable")
- Bash tool heredocs on this Windows setup collapse backslash escapes (a `"\\n"` typed inside a Python string in the heredoc reached the file as a real newline, `"\\r"` as a raw CR) in Python code written to files; use the Edit tool for lines with escapes, or build them with chr(92) (verified 2026-11-20: report.py return line split across lines, test_report.py assert held a raw CR)
