"""Project a pinned validation-only report into a small static UI dataset."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'work/validation_sensitivity_20261012/validation_sensitivity.json'
EXPECTED = 'b5ec1fd924f01063b4b17bb643e272ec9eaf8c13d9825c42f58528a632273d12'
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED
report = json.loads(raw)
assert report['scope'] == 'DESCRIPTIVE_VALIDATION_PARAMETER_SENSITIVITY_ONLY'
assert report['counts']['groups'] == 1300 and report['new_backtests'] == 0
groups = {}
for row in report['groups']:
    groups[row['group']] = {
        'attempts': row['observed_attempts'], 'scientific_points': row['observed_scientific_points'],
        'aliases': row['cache_alias_attempts'], 'missing': row['missing_scientific_points'],
        'metrics': row['metric_distribution_scientific_points'],
        'axes': [{
            'axis': a['axis'], 'active': a['axis_active'],
            'comparisons': a['comparison_status_counts'].get('EFFECTIVE_NEIGHBOR', 0),
            'net_delta_median': a['metric_high_minus_low']['net_profit']['median'],
            'net_abs_delta_median': a['metric_absolute_delta']['net_profit']['median'],
            'sign_reversals': a['strict_net_sign_reversals'],
        } for a in row['axes']],
    }
payload = {'schema_version': 1, 'scope': 'VALIDATION_ONLY',
           'campaign_id': 'single-frame-8af790ef59a5', 'source_report_sha256': EXPECTED,
           'source_csv_sha256': report['input']['sha256'], 'axes': report['axes'], 'groups': groups}
target = ROOT / 'frontend/public/research-audits/validation-sensitivity-20261012.json'
target.parent.mkdir(parents=True, exist_ok=True)
data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
with target.open('xb') as f:
    f.write(data)
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
receipt = {'source': str(SOURCE), 'source_sha256': EXPECTED, 'output': str(target),
           'output_sha256': hashlib.sha256(data).hexdigest(), 'groups': len(groups),
           'new_statistics_computed': False, 'selection_or_qualification_changed': False,
           'transformation': 'Exact projection of saved group distributions and axis summaries; JSON encoding only'}
with (ROOT / 'work/sensitivity_ui_projection_20261012.json').open('x') as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'groups': len(groups), 'bytes': len(data), 'sha256': receipt['output_sha256']}))
