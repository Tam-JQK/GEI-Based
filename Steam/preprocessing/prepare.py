"""Build a new processed dataset from its raw interaction and item tables."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New directory for processed data')
    args = parser.parse_args()
    dataset = json.loads((ROOT / 'configs/protocol.json').read_text())['dataset']
    if dataset == 'BeerAdvocate':
        import repartition_multi as preparation
        build = preparation.process
    else:
        import repartition as preparation
        build = preparation.prepare

    from analysis.artifacts import file_hash

    raw = ROOT / 'data/raw' / dataset
    expected = json.loads((raw / 'SHA256SUMS.json').read_text())
    for name, checksum in expected.items():
        path = raw / name
        if not path.is_file():
            parser.error(f'Missing raw input: {path}')
        if file_hash(path) != checksum:
            parser.error(f'Raw input differs from the recorded snapshot: {path}')

    preparation.OUT = args.output.resolve()
    preparation.REPORT = preparation.OUT / 'reports'
    destination = preparation.OUT / dataset
    if destination.exists():
        parser.error(f'Choose a new output path: {destination} already exists')
    preparation.OUT.mkdir(parents=True, exist_ok=True)
    preparation.REPORT.mkdir(parents=True, exist_ok=True)
    build(dataset)
    config_path = destination / 'process_config.yaml'
    config_path.write_text(''.join(line for line in config_path.read_text().splitlines(keepends=True)
                                   if not line.startswith('processed_data_dir:')))
    print(f'Processed data written to {destination}')


if __name__ == '__main__':
    main()
