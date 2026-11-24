"""Explicit new 1h training. Preserves all 24h artifacts and news workflows."""
import argparse
import json
from one_hour_trend import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Use the separate 1h raw cache; completed runs are reused without rescoring.')
    args = parser.parse_args()
    try:
        report = train(offline=args.offline)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, '1-hour research stopped safely: ' + str(exc) + '\n')
    print(json.dumps({key: report[key] for key in ('version', 'selected_model', 'threshold_pct', 'split', 'test', 'baselines', 'historical_quality')}, indent=2))
