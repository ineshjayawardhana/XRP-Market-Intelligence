"""Run the isolated model-performance investigation, without production writes."""
import argparse
import json
from pathlib import Path

from trend_experiments import ROOT, run_investigation
from trend_experiment_report import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Use isolated experiment cache without network')
    parser.add_argument('--development-only', action='store_true', help='Never access reserved final targets')
    args = parser.parse_args()
    try:
        report = run_investigation(ROOT, args.offline, args.development_only)
        paths = write_report(report, ROOT)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, 'Investigation stopped safely: ' + str(exc) + '\n')
    print(json.dumps(dict(selected=report['selected_group'], final_metrics=report.get('final_metrics'),
        quality=report.get('quality'), consistency=report.get('consistency'), report_files=paths,
        preservation=report['preservation']), indent=2))


if __name__ == '__main__':
    main()
