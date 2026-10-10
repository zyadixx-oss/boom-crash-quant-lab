# Execution notes

- The source was the new official public Deriv channel, not the older third-party
  SQLite database. Exactly 180 elapsed days were retrieved for each symbol.
- The literal bootstrap seed description in the frozen protocol is shorthand:
  the implementation uses base seed 20261004 **plus block_hours**, hence 20261028
  for UTC days and 20261010 for six hours. No outcomes determined the seed.
- The first attempted run found zero eligible outcomes because Pandas 3 native
  timestamps used seconds/microseconds while the comparison expected nanoseconds.
  That run was invalid. Timestamp normalization was corrected; the successful
  run has 36,036 development and 15,546 final eligible M5 opportunities per symbol.
- Adversarial tests also found cooldown reservation across overlapping hourly
  references. Applying cooldown in chronological signal issue order corrected it.
- The original acceptance test searched for a lowercase environment variable
  although all actual environment flags are uppercase. It now checks all four
  false defaults. The safety guard still refuses any flag enabled by environment
  or repository .env.
- The main 384 final comparisons and all walk-forward rows were executed using
  the frozen numeric signal rules. UTC-day bootstrap has 9999 replicates.
- Eight research-gate passes belong to the **same Crash500 SR_ALIGNMENT model**
  under eight related excursion definitions. They are not eight independent
  strategies or evidence of profitable trading. Their original minimum-resolved
  p=0.0001 gives Holm p=0.0384; a finer resampling sensitivity is documented
  separately. All intervals are pointwise and inference is approximate.
- After observing the main results, an explicit post-hoc ATR-decile matched
  baseline diagnostic was added to separate timing information from easier
  volatility-normalized outcomes. Its findings cannot count as a fresh test.
- Raw and normalized CSVs are preserved locally and in the deliverable bundle;
  they are gitignored. GitHub holds summaries, metrics and acquisition/audit
  manifests. No source prices were interpolated, and missing windows are rejected.
- All data access was public. No real/demo orders, account login, balances or
  authorization tokens were used. Four live flags remain false.
