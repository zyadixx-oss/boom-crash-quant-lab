from pathlib import Path
import json, hashlib
from decimal import Decimal, localcontext
from fractions import Fraction
from datetime import datetime
from collections import defaultdict
root = Path.cwd()
folder = root / 'docs/provider_evidence_20261009'
cap = root / 'docs/public_quote_contract_20261007/capture01'
checks = 0
def require(condition, message):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(message)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def unique(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result
def bad_constant(value):
    raise ValueError('Non-finite JSON constant')
def parse(text):
    return json.loads(text, parse_float=Decimal, object_pairs_hook=unique, parse_constant=bad_constant)
inv = parse((folder / 'inventory.json').read_text())
original_result = parse((cap / 'result.json').read_text())
original_declaration = parse((cap / 'declaration.json').read_text())
require(inv['source_path'] == 'docs/public_quote_contract_20261007/capture01/received_text_messages.jsonl', 'Exact source path')
raw = cap / 'received_text_messages.jsonl'
require(digest(raw) == inv['source_sha256'] == 'afc224883af67d1730ec09de99d3655ecabe7ea4cf6d63fd0a44cdbc9b59b298', 'Raw source pin')
require(digest(folder / 'review_inventory.py') == inv['review_source_sha256'] == '72dec4854b14409ef18ed1eb1c8e2dc56a6df08d92c2a8f796deab09de60abc6', 'Inventory script pin')
for filename, expected in original_result['artifact_sha256'].items():
    require(digest(cap / filename) == expected, 'Original artifact pin: ' + filename)
require(digest(cap / 'probe_source.py') == original_declaration['source_sha256'], 'Original probe source pin')
require(digest(cap / 'probe_tests.py') == original_declaration['tests_sha256'], 'Original probe test pin')
prior_review = parse((cap.parent / 'capture01_review.json').read_text())
require(digest(cap / 'result.json') == prior_review['capture_result_sha256'] == '745ffffacc2b0f17ae6696e36ee9489f00baa2053252a622684ec4e26caa429e', 'Prior reviewed capture result')
require(digest(cap.parent / 'review_capture01.py') == prior_review['source_sha256'], 'Prior independent reviewer source pin')
require(original_result['status'] == prior_review['status'] == 'PASS', 'Original capture/review status')
flags = {'LIVE_TRADING','READY_FOR_LIVE','LIVE_ALLOWED','OPENED_TRADES'}
for document in (inv, original_result, original_declaration):
    require(set(document['safety']) == flags, 'Exact safety flag keys')
    for name in flags:
        require(document['safety'][name] is False, 'False safety flag ' + name)
expected_false = ['rounding_probe_is_predictive_test', 'all_public_endpoints_exhaustively_reviewed', 'new_predictor_established', 'universal_profit_impossibility_proven', 'executable_CFD_mapping_verified', 'historical_costs_verified', 'profit_tested_by_this_inventory', 'QUALIFIED']
for key in expected_false:
    require(inv[key] is False, 'Limited scope: ' + key)
require(inv['rounding_probe_chosen_after_capture_observed'] is True, 'Explicitly posthoc')
require(inv['posthoc_fixed_half_markup'] == '0.00000675', 'Fixed descriptive rule')
require(inv['stage'] == 'descriptive_saved_public_field_inventory', 'Descriptive inventory stage')
groups = defaultdict(list)
for position, line in enumerate(raw.read_text().splitlines(), 1):
    envelope = parse(line)
    require(envelope['frame_number'] == position, 'Frame sequence')
    require(envelope['payload_type'] == 'text', 'Text payload')
    require(hashlib.sha256(envelope['text'].encode()).hexdigest() == envelope['payload_sha256'], 'Payload SHA')
    obj = parse(envelope['text'])
    tick = obj['tick']
    require(obj['msg_type'] == 'tick', 'Tick response')
    require(tick['id'] == obj['subscription']['id'], 'Tick ID equals subscription identity')
    require(set(tick) == {'ask','bid','epoch','id','pip_size','quote','symbol'}, 'Exact tick fields')
    require(type(tick['epoch']) is int and type(tick['pip_size']) is int, 'Integer epoch and precision')
    require(all(isinstance(tick[k], (int, Decimal)) and not isinstance(tick[k], bool) for k in ('quote','bid','ask')), 'Numeric price schema')
    require(0 < tick['bid'] <= tick['quote'] <= tick['ask'], 'Positive ordered quote')
    groups[tick['symbol']].append(tick)
require(set(groups) == set(inv['symbols']) == {'BOOM500','CRASH500','BOOM600','CRASH600'}, 'Exact symbol population')
require(sum(map(len, groups.values())) == inv['messages'] == 240, 'Total rows')
def rational_round_even(value):
    whole, remainder = divmod(value.numerator, value.denominator)
    twice = remainder * 2
    return whole + (twice > value.denominator or (twice == value.denominator and whole % 2 == 1))
observed = {}
for symbol, ticks in sorted(groups.items()):
    require(len(ticks) == 60, 'Per-symbol rows')
    epochs = [row['epoch'] for row in ticks]
    require(epochs == list(range(1791372970,1791373030)), 'Exact contiguous epoch window')
    matches = 0
    ratios = []
    for row in ticks:
        scaled = Fraction(row['quote']) * 10 ** row['pip_size']
        lower = rational_round_even(scaled * Fraction(3999973,4000000))
        upper = rational_round_even(scaled * Fraction(4000027,4000000))
        matches += (Fraction(row['bid']) * 10 ** row['pip_size'] == lower and Fraction(row['ask']) * 10 ** row['pip_size'] == upper)
        with localcontext() as ctx:
            ctx.prec = 28
            ratios.append((Decimal(row['ask']) - Decimal(row['bid'])) / Decimal(row['quote']))
    expected = {
        'messages':len(ticks),
        'observed_tick_key_sets':[sorted({'ask','bid','epoch','id','pip_size','quote','symbol'})],
        'distinct_subscription_ids':len({row['id'] for row in ticks}),
        'first_epoch':epochs[0], 'last_epoch':epochs[-1],
        'spread_over_quote_min':str(min(ratios)),
        'spread_over_quote_max':str(max(ratios)),
        'posthoc_fixed_half_markup_exact_matches':matches,
    }
    require(set(inv['symbols'][symbol]) == set(expected), 'Exact saved symbol fields')
    for key, value in expected.items():
        require(inv['symbols'][symbol][key] == value, symbol + ': ' + key)
    observed[symbol] = {'rows':len(ticks), 'matches':matches, 'distinct_subscription_ids':expected['distinct_subscription_ids']}
require(datetime.fromisoformat(inv['reviewed_utc']) > datetime.fromisoformat(original_result['finished_utc']), 'Review follows capture')
print(json.dumps({'status':'PASS','independent_checks':checks,'raw_payload_hashes_verified':240,'original_artifact_hashes_verified':len(original_result['artifact_sha256']),'inventory_sha256':digest(folder/'inventory.json'),'review_script_sha256':digest(folder/'review_inventory.py'),'symbols':observed,'method':'stdlib only; exact Fraction half-even rounding independent of production Decimal quantize; Decimal28 ratios','scope':'saved bytes/schema/descriptive arithmetic and flags; no official page re-fetch or predictive/economic validation'}, indent=2))
