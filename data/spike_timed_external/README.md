# Timed-exit v2 external data acquisition — PARTIAL

Only public active_symbols and ticks_history were called. All four safety flags are false. No outcomes were inspected.

Requested180days: [2025-04-12 11:08UTC,2025-10-09 11:08UTC). The API returned candles later than the requested end at its historical boundary, so180days are unavailable through this channel. The initial collector aborted; collector_partial.py records that boundary and preserves already fetched authentic rows. Its modifications and hashes are in collector_provenance.json.

| Symbol | Rows | Actual first UTC | End exclusive UTC | Internal gaps | Overlap with v1 |
|---|---:|---|---|---:|---:|
| BOOM500 | 6170 | 2025-10-05T04:18:00+00:00 | 2025-10-09T11:08:00+00:00 | 0 | 0 |
| CRASH500 | 6170 | 2025-10-05T04:18:00+00:00 | 2025-10-09T11:08:00+00:00 | 0 | 0 |

Both instruments have6170minutes (4days6hours50minutes), approximately2.38% of the requested180days;253030 earlier minutes are unavailable. This is a short independent historical interval, not180days or a sufficiently powered confirmatory cohort.

Each raw file contains6171rows; one boundary candle starts at2025-10-05 04:17:26UTC and is excluded as off-grid. The normalized files contain6170canonical complete-minute rows, with finite positive OHLC, valid bounds and no duplicates. Raw and normalized hashes therefore differ, as recorded in the normalization manifests. The original v1 sources/protocol remain unchanged. Full response hashes and exact requested/returned boundary timestamps are in the page audits.
