# Scope and repair history

This is a public capability investigation prompted by a documentation lead,
not another fit against exposed returns. The economic objective remains unmet.
No previous economic source/result/acceptance gate was changed.

Before the first observation:

1. Initial 20 local cases passed; independent source review found that an
   absent/malformed proposal ID was incorrectly described as SUCCESS and a
   huge integer multiplier raised OverflowError rather than becoming unknown.
   The added regressions produced five failed ID subcases and one error.
   `premeasurement_source_01.py` and `premeasurement_self_test_01.log` preserve
   this state. Both defects were fixed; 21 cases passed.
2. Independent review of installed websockets 17.2 found automatic HTTP
   redirect following. An added regression failed before PublicConnect was
   implemented. Source02 and log02 preserve this; 22 cases then passed.
   The declared deadline was clarified as 300 seconds request work plus up
   to 5 seconds close cleanup before any live public observation.
3. Python dictionary equality accepts True == 1. A response-echo regression
   exposed that issue; source03 and log03 preserve it. Canonical JSON echo
   equality now distinguishes types. Final freeze: 22 cases passed.

Version 1 source/test/protocol SHA256:

- probe.py: e4f23509b1cc55540a4d9bfd9e5031743f1c137f8dfcda625d2418056becf608
- test_probe.py: 64b28627066db6a17476ff89465ad67f36709215798466492a37a4bebbe169cc
- PROTOCOL.md: 8ef504678c16b673c4060d448d53b739bd7724e58606dd03f71e735ed2bb23cd

After capture01 stopped on an echo mismatch, separate v2 files were made.
The previously archived official request schema defines duration_unit default
"s". Only the added default is accepted, only for MULTUP/MULTDOWN, alongside
the original exact echo. No request specifications or economic assumptions
changed. 23 cases passed before capture02. Independent read-only review
recomputed all six v1/v2 pins and confirmed this narrow difference and the
installed redirect override behavior. Its source review did not rerun tests,
inspect TLS traffic, or verify economic performance.

Version 2 source/test/protocol SHA256:

- probe_v2.py: 7bb49f36833822374675658de36c0e04e0988e4c2d76ccc35c5739f294abbe14
- test_probe_v2.py: 0e1ae028ba1779db87a019edaebe5e026cdea7dda3816641dc3ba4dbdd527bc0
- PROTOCOL_v2.md: 0a6d255c680fdeb2f51f76c2112f75c180caa998ab185e4ad68bedb29d90ae50

All captured source copies and raw messages remain immutable. API errors,
unknowns, skips and the incomplete first capture must remain visible in
future summaries. The archived master schemas are timestamped byte snapshots,
not a guarantee that the implementation conforms to every schema field.
In particular, growth_rate_barrier_offsets was observed but absent from the
saved contracts_for response schema. No positional/unit mapping or forecast
was inferred from it.

compare_content.py independently of the probe reads only the two saved raw
message files and the prior saved stable-objective assessment. It reproduces
the descriptive quote table, five two-observation offset equalities and the
18 positive model/rule rows with zero necessary conjunction passes. It is
root-authored, not an independent economic or capture audit.

Final independent offline capture review: 210 checks for capture01 with no
consistency errors, evidence_consistent=true, status=INCOMPLETE, pass=false;
427 checks for capture02 with no errors and PASS for capability consistency.
Sixteen synthetic cases passed, including end-to-end raw/pin/matrix/summary
tampering, missing artifacts, unsafe flags, added orders, changed echo policy,
unknown proposal IDs and exact incomplete/complete classification. This is
not an independent network, execution, predictive or economic audit.

Reviewer code SHA256:
8af9f223b18543031ea8359cc5a16ec33beb419973cd11d472caf010fdbfb0f1

Saved review01 SHA256:
cffe337653640fc46aa0ffdcbd3d5ef7227ab0ce0fb3fa6abc5afa6adb396136

Saved review02 SHA256:
f4c9328e43eabdb61cfa87d9a849120e275a39139ebc8662b0400fa71a3448c5
