"""Explicit training entry point; dashboard never calls this script."""
import argparse
import json
from pathlib import Path

from trend_pipeline import ROOT, train_research


def main():
    parser = argparse.ArgumentParser(description='Purged chronological XRP 24h trend research')
    parser.add_argument('--days', type=int, default=730, help='History days, 180-1825 (default730). More than six months helps independent evaluation.')
    parser.add_argument('--offline', action='store_true', help='Use previously collected data/trend/xrp_hourly.csv; no network.')
    parser.add_argument('--root', type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        report = train_research(args.root, args.days, args.offline)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, 'Trend research stopped safely: ' + str(exc) + '\n')
    print(json.dumps(dict(model_version=report['model_version'], selected_model=report['selected_model'],
        data=report['data'], split=report['split'], holdout=report['holdout'], quality=report['quality'],
        news_comparison_available=report['news']['ready']), indent=2))


if __name__ == '__main__':
    main()
