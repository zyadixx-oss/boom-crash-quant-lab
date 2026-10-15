# Actual public quote-contract observation — 7 October 2026

**This capture verifies public-feed bid/ask availability, not executable CFD
costs or strategy profitability.** It changes the earlier broad availability
assumption: the historical research files are quote-only, while the current
public stream can include bid and ask.

A bounded unauthenticated capture ran2026-10-07 11:36:07.855–11:37:09.678UTC at
`wss://api.derivws.com/trading/v1/options/ws/public`. Four allowlisted `ticks`
requests used `subscribe:1`. The socket closed successfully after240 decoded
text messages:60 distinct consecutive epochs per symbol, with no missing
seconds, duplicates, unknown sides or ordering failures in this short sample.
Raw decoded message text, per-message hashes, sent requests, local receipt times,
source/test snapshots and runtime metadata are saved in [capture01](capture01/).
WebSocket framing/TLS bytes and actual order-processing latency were not captured.

| Symbol | Observations | Min public spread | Median | Max |
|---|---:|---:|---:|---:|
| BOOM500 | 60 | 0.077 | 0.078 | 0.078 |
| CRASH500 | 60 | 0.040 | 0.041 | 0.041 |
| BOOM600 | 60 | 0.068 | 0.069 | 0.069 |
| CRASH600 | 60 | 0.315 | 0.315 | 0.315 |

All spreads are **price units of this public feed**, not dollars, pip values,
round-trip CFD fees or historical costs. Sixty seconds per symbol cannot supply
a representative long-period spread distribution.

The [independent stdlib raw-message review](capture01_review.json) rechecked all
240 payload hashes and bid<=quote<=ask relationships, the four exact public
requests, saved observations, distinct contiguous epochs and decimal spread
summaries. [Review source](review_capture01.py) is executable without network.
The live probe's52 synthetic tests passed; a pytest reserved-parameter collection
failure is preserved as a transcript-derived record before repair. Exploratory
API attempts are separately labeled [transcript observations](exploratory_transcript_observations.json);
their original wire bytes were not retained and they are not primary capture
evidence. The controlled capture and review both passed on their first actual run.

The official [tick schema](https://raw.githubusercontent.com/deriv-com/deriv-api-schemas/master/schemas/ticks_response.schema.json)
describes bid/ask fields, while [historical tick examples](https://developers.deriv.com/comparison/ticks-history/)
return prices/times. The [migration page](https://developers.deriv.com/comparison/ticks/)
describes subscribe0 as available, but exploratory requests at this options
public endpoint rejected it. The controlled collector therefore uses explicit
subscribe1 and closes the socket rather than assuming one-shot behavior.

No public-feed-to-MT5 executable CFD mapping, measured fill, commission,
financing or execution latency was established. Public [CFD specifications](https://deriv.com/trading-specifications)
list indicative contract/spread/swap fields but qualify their currency and
availability; they cannot replace historical account-specific executions.
Do not insert these current public spreads into old quote-only paths or present
them as a measured replacement for the existing0.10ATR assumption.

The separate [fixed-ledger cost ceiling](../frozen_cost_ceiling_20261007/REPORT.ar.md)
checks whether reducing additive costs alone could rescue existing models.
This market-data observation itself neither changes a model nor produces a
strategy candidate. Money profit and prospective paper results remain NOT TESTED.

`LIVE_TRADING=false`, `READY_FOR_LIVE=false`, `LIVE_ALLOWED=false`,
`OPENED_TRADES=false`. No accounts, credentials or orders were used.
