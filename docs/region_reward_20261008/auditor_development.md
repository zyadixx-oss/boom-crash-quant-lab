# Independent auditor development evidence

The first synthetic test execution returned exit 1: 3 failed, 70 passed.
Two mirrored stop fixtures unintentionally left the quote immediately after
entry at 100, causing an earlier valid stop than the intended synthetic scenario.
The fixture now retains its entry level until the intended stop trigger.
One paired-model hash test exposed inconsistent pandas datetime resolution:
the independent target hash now explicitly converts issue timestamps to
nanoseconds before encoding big-endian int64 values, matching the declared
UTC nanosecond contract.

After these repairs, 73 tests passed and both owned Python files compiled.
No production runner, learner, protocol, declaration or result was changed.
The actual historical audit then passed on its first execution, exit 0, with
2,300,121 checks and no errors. No failed historical audit was replaced.
The stdout transcript records observed tool output chunks from session 63247;
it was saved after completion, not captured by redirecting the live process.
