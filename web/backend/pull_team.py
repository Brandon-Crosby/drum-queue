"""Download the beta team's kept splits from Modal to your Mac.

    bash -c '$(head -1 "$(which modal)" | cut -c3-) web/backend/pull_team.py'    # into ~/DrumSplits/team
(runs it with the Python the modal command uses; add --to SOME/DIR inside the quotes to change the folder)

Only new files are downloaded, so run it as often as you like. Each split arrives as
'<date> <id> <song>/' with input/ (the upload), out/<song>/ (stems, one-shots, groove MIDI) and job.json.
(File by file on purpose: `modal volume get` on a folder with subfolders wrote everything onto one file.)
"""
import argparse
from pathlib import Path

import modal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--to', default=str(Path.home() / 'DrumSplits' / 'team'))
    a = ap.parse_args()
    dest = Path(a.to).expanduser()
    vol = modal.Volume.from_name('drumsplit-data')
    n = skipped = 0
    try:
        entries = list(vol.iterdir('/team', recursive=True))
    except modal.exception.NotFoundError:
        print('Nothing kept yet: no split has come through the team link so far.')
        return
    for e in entries:
        if e.type != modal.volume.FileEntryType.FILE:
            continue
        rel = e.path.lstrip('/')
        rel = rel[len('team/'):] if rel.startswith('team/') else rel
        out = dest / rel
        if out.exists() and out.stat().st_size == e.size:
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'wb') as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
        n += 1
        print(f'  {rel}')
    print(f'{n} new file(s) -> {dest}' + (f' ({skipped} already here)' if skipped else ''))


if __name__ == '__main__':
    main()
