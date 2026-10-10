# Frozen public contract capability audit

This declaration precedes the first WebSocket request in this audit. It tests
public data availability only, not a strategy, conditional probability, CFD
execution, costs, fills, or profitability. It does not change the accepted
PF > 1 / positive stable net-R objective or reuse exposed periods as fresh OOS.

- Endpoint: `wss://api.derivws.com/trading/v1/options/ws/public` only.
- Symbols, in order: BOOM500, CRASH500, BOOM600, CRASH600; R_100 is a
  public Volatility control, outside the economic strategy's scope.
- First request the exact `contracts_for` catalogue once per symbol.
- For each successful catalogue, inspect only CALL, PUT, MULTUP, MULTDOWN,
  in that order. No requests for other products or arbitrary parameters.
- CALL/PUT: quote only if an advertised zero-barrier record explicitly has
  tick expiry, unit-bearing minimum/maximum durations enclosing 5t. Request
  duration=5, duration_unit=t. Bare duration strings remain unknown.
- MULTUP/MULTDOWN: quote only if an advertised record has a nonempty array
  of finite, strictly positive numeric multipliers. Copy its lowest listed
  value directly. Different advertised records are preserved; this rule
  does not infer an interval between listed values. Conflicting or malformed
  multiplier lists for the same symbol/type skip that product.
- Each quote has proposal=1, amount=10, basis=stake, currency=USD,
  underlying_symbol and a fixed request ID. No subscribe field, limit_order,
  cancellation, buy, sell, account, authentication, or token fields.
- A quote is informational and cannot open a position. No returned proposal
  ID may be used for another operation. No currency/account negotiation.
- All four safety flags must be false. Reject conflicting environment values.
- Maximum one connection, 25 requests/responses, 300 seconds request work
  plus up to 5 seconds for connection cleanup,
  15 seconds per receive/send, 20 seconds connection, 5 seconds close.
- No retries, endpoint fallback, or changed parameters after seeing a result.
  HTTP redirects are explicitly disabled in the WebSocket connector.
  Errors remain exact observations. A catalogue error skips its proposals.
  Empty/unrecognized metadata skips only the relevant proposals with a reason.
- Preserve exact decoded response text and SHA256, sent request text,
  timestamps, declaration/source/test hashes, runtime and final summary.
  Require type-sensitive request-ID and canonical-JSON echo equality.
  TLS frames and network packets are not captured.
- Preserve every received field. Missing optional quote values stay unknown.
  A missing, null, empty or non-string proposal ID is UNUSABLE_PROPOSAL;
  preserve that reply and continue the fixed matrix without changing it.
  Do not require all legacy response fields, fill missing prices with zero,
  equate Options quotations with historical/executable CFD prices, or infer
  hidden generator state merely from field names.
- Accumulators and any other advertised types are catalogue observations
  only in this audit; their price proposals are NOT TESTED.
- Successful quotes would justify a separate data-content question, not a
  forecast or profitable strategy. Failure applies only to these exact
  unauthenticated requests, symbols, endpoint and observation time.

Official references read before measurement:

- https://developers.deriv.com/docs/options/ws-public/
- https://developers.deriv.com/docs/data/contracts-for/
- https://developers.deriv.com/docs/trading/proposal/
- https://developers.deriv.com/comparison/contracts-for/
- https://developers.deriv.com/comparison/proposal/
- https://github.com/deriv-com/deriv-api-schemas/tree/master/schemas

The current comparison says only proposal.id is required and pricing fields
may be strings; raw master response schemas can differ. Neither source
overrides the exact preserved endpoint response.
