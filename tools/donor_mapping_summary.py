#!/usr/bin/env python3
"""Validate all donor assignments and regenerate the compact sensitivity table."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

MODELS = (20230601, 20230602, 20230603)
MAPS = (20260921, 20260922, 20260923, 20260924, 20260925)
SCOPES = {'latest_1': (87044, 2843), 'all': (66139, 2170)}


def summarize(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    expected = {(m, s, scope) for m in MAPS for s in MODELS for scope in SCOPES}
    identities = [(int(r['mapping_seed']), int(r['model_seed']), r['scope']) for r in rows]
    if len(rows) != 30 or len(set(identities)) != 30 or set(identities) != expected:
        raise ValueError('Expected all five assignments, three checkpoints and two scopes')
    for r in rows:
        n, users = SCOPES[r['scope']]
        if (int(r['rows']), int(r['user_clusters'])) != (n, users):
            raise ValueError('Support denominator changed')
        if float(r['ci_level']) != .975 or int(r['bootstrap_iters']) != 10000:
            raise ValueError('Uncertainty protocol changed')
        for prefix in ('c2', 'paired_c0_minus_c2'):
            point, low, high = [float(r[prefix + suffix]) for suffix in ('_delta_rr', '_ci_low', '_ci_high')]
            if not all(math.isfinite(x) for x in (point, low, high)) or low > high:
                raise ValueError('Invalid estimate or interval')
            status = 'positive' if low > 0 else 'negative' if high < 0 else 'unresolved'
            if r[prefix + '_status'] != status:
                raise ValueError('Interval status differs from bounds')
        if abs(float(r['c0_delta_rr']) - float(r['c2_delta_rr']) - float(r['paired_c0_minus_c2_delta_rr'])) > 1e-12:
            raise ValueError('Paired contrast identity failed')
    result = []
    for seed in MODELS:
        for scope in SCOPES:
            group = [r for r in rows if int(r['model_seed']) == seed and r['scope'] == scope]
            c0 = [float(r['c0_delta_rr']) for r in group]
            if max(c0) - min(c0) > 1e-12:
                raise ValueError('C0 changed with donor assignment')
        group = [r for r in rows if int(r['model_seed']) == seed and r['scope'] == 'all']
        c2 = [float(r['c2_delta_rr']) * 1000 for r in group]
        gap = [float(r['paired_c0_minus_c2_delta_rr']) * 1000 for r in group]
        result.append({'checkpoint': group[0]['checkpoint'], 'model_seed': seed,
                       'assignments': len(group), 'c2_min': min(c2), 'c2_max': max(c2),
                       'paired_min': min(gap), 'paired_max': max(gap),
                       'paired_positive_intervals': sum(float(r['paired_c0_minus_c2_ci_low']) > 0 for r in group),
                       'c2_positive_intervals': sum(float(r['c2_ci_low']) > 0 for r in group)})
    return result


def table_lines(summary: list[dict], mark: bool = False) -> tuple[list[str], list[str]]:
    def wrap(value):
        return r'\rev{' + value + '}' if mark else value
    heading = '(c) Five alternative C2 assignments: all-source point-estimate ranges'
    header = [r'\toprule', r'\multicolumn{4}{l}{' + wrap(heading) + r'} \\',
              ' & '.join(wrap(x) for x in ['Checkpoint', 'C2 range', 'C0--C2 range', r'Paired 97.5\% CIs above zero']) + r' \\', r'\midrule']
    body = []
    for r in summary:
        values = [r['checkpoint'], f"${r['c2_min']:+.3f}\\ \\text{{to}}\\ {r['c2_max']:+.3f}$",
                  f"${r['paired_min']:+.3f}\\ \\text{{to}}\\ {r['paired_max']:+.3f}$",
                  f"${r['paired_positive_intervals']}/{r['assignments']}$"]
        body.append(' & '.join(wrap(x) for x in values) + r' \\')
    return header, body


def render_panel(summary: list[dict], mark: bool = False) -> str:
    header, body = table_lines(summary, mark)
    return '\n'.join([r'\vspace{4pt}', r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c c c@{}}',
                      *header, *body, r'\bottomrule', r'\end{tabular*}']) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-summary', type=Path)
    parser.add_argument('--output-tex', type=Path)
    parser.add_argument('--mark-changes', action='store_true')
    args = parser.parse_args()
    summary = summarize(args.input)
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        with args.output_summary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    if args.output_tex:
        args.output_tex.parent.mkdir(parents=True, exist_ok=True)
        args.output_tex.write_text(render_panel(summary, args.mark_changes), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'conditions': 30, 'all_source_summary': summary}, indent=2))


if __name__ == '__main__':
    main()
