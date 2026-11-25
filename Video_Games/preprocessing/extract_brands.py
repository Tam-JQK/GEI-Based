"""Extract same-ASIN brand metadata from Amazon Reviews 2023."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import re

import pandas as pd

from repartition import clean

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'Amazon_Video_Games'


def brand(record):
    fields = record.get('details') or {}
    if not isinstance(fields, dict):
        fields = {}
    entries = [value for key, value in fields.items()
               if re.sub(r'[^a-z]', '', key.casefold()) == 'brand']
    if record.get('brand'):
        entries.append(record['brand'])
    names = {}
    for entry in entries:
        for value in entry if isinstance(entry, list) else [entry]:
            label = clean(value)
            if label:
                names[label.casefold()] = label
    return next(iter(names.values())) if len(names) == 1 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True, help='Official meta_Video_Games.jsonl.gz')
    parser.add_argument('--output', type=Path, required=True, help='Brand TSV for the input item catalog')
    args = parser.parse_args()
    items = ROOT / 'data/raw' / DATASET / (DATASET + '.item')
    meta = pd.read_csv(items, sep='\t', dtype=str, usecols=['item_id:token', 'title:token']).dropna()
    targets = set(meta['item_id:token'])
    pattern = re.compile(rb'"parent_asin"\s*:\s*"([^"]+)"')
    brands = defaultdict(dict)
    with gzip.open(args.metadata, 'rb') as stream:
        for line in stream:
            match = pattern.search(line)
            if match is None:
                continue
            asin = match[1].decode()
            if asin in targets:
                label = brand(json.loads(line))
                if label:
                    brands[asin].setdefault(label.casefold(), label)
    with args.output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(['item_id', 'supplier'])
        for asin, names in brands.items():
            if len(names) == 1:
                writer.writerow([asin, next(iter(names.values()))])


if __name__ == '__main__':
    main()
