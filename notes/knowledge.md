# Knowledge

Facts about this repo and its environment that took effort to find out. The agent adds a line here when it needed
3 or more tries to find the right way, or when you corrected a wrong claim about the repo.
Line format: `- <fact>; <the right way> (verified YYYY-MM-DD: <command or evidence>)`.
Not here: bugs (dbg journal), decisions (decisions.md), agent mistakes (FAILURES.md).

- Two verify scripts read .agentkit.json differently: .agentkit/scripts/verify.py (hook, unified keys verify.tests) and the skill vf .agentkit/skills/verifying-changes/scripts/verify.py (reads verify.test only, else probes system python, which lacks coverage, and skips tests); keep both verify.test and verify.tests until the skill copy is upgraded, and check with `vf --detect` that the test line says "set in .agentkit.json" (verified 2026-11-19: vf --detect after dropping verify.test showed "pytest not runnable")
