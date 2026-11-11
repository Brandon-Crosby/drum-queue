#!/usr/bin/env python3
"""Render web/site/ into web/dist/ for Cloudflare Pages.

  python3 web/build_site.py

Fills {{KEYS}} from site.json (environment variables of the same name win, so Cloudflare's
dashboard can override them), drops in the shared header/footer, and writes ads.txt,
robots.txt and sitemap.xml. No dependencies beyond Python 3.
"""
import json
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / 'site', HERE / 'dist'
TEXT = {'.html', '.js', '.css', '.txt', '.xml', '.json', '.svg'}


def main():
    cfg = json.loads((HERE / 'site.json').read_text())
    cfg = {k: os.environ.get(k, v) for k, v in cfg.items()}
    ads = cfg.get('ADSENSE_CLIENT', '')
    cfg['ADSENSE_TAG'] = (
        f'<meta name="google-adsense-account" content="{ads}">\n'
        f'<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={ads}" '
        f'crossorigin="anonymous"></script>') if ads else ''
    ts = cfg.get('TURNSTILE_SITEKEY', '')
    cfg['TURNSTILE_TAG'] = ('<script src="https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit" '
                            'async defer></script>') if ts else ''
    partials = {p.stem.upper(): p.read_text() for p in (SRC / '_partials').glob('*.html')}

    def fill(text):
        for _ in range(2):                      # partials can contain {{KEYS}} too
            text = re.sub(r'\{\{(\w+)\}\}', lambda m: partials.get(m[1], cfg.get(m[1], m[0])), text)
        return text

    shutil.rmtree(OUT, ignore_errors=True)
    pages = []
    for f in SRC.rglob('*'):
        rel = f.relative_to(SRC)
        if f.is_dir() or rel.parts[0] == '_partials':
            continue
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix in TEXT:
            dst.write_text(fill(f.read_text()))
        else:
            shutil.copy2(f, dst)
        if f.suffix == '.html' and f.name != '404.html':
            path = rel.as_posix().removesuffix('.html')
            pages.append(path.removesuffix('index') if path.endswith('index') else path)

    left = re.findall(r'\{\{\w+\}\}', ' '.join(p.read_text() for p in OUT.rglob('*.html')))
    if left:
        raise SystemExit(f'Unfilled placeholders: {sorted(set(left))}')

    base = f"https://{cfg['DOMAIN']}"
    (OUT / 'robots.txt').write_text(f'User-agent: *\nAllow: /\nDisallow: /api/\nSitemap: {base}/sitemap.xml\n')
    (OUT / 'sitemap.xml').write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + ''.join(f'  <url><loc>{base}/{p}</loc></url>\n' for p in sorted(pages))
        + '</urlset>\n')
    if cfg.get('ADSENSE_CLIENT'):
        pub = cfg['ADSENSE_CLIENT'].removeprefix('ca-')
        (OUT / 'ads.txt').write_text(f'google.com, {pub}, DIRECT, f08c47fec0942fa0\n')
    print(f'Built {len(pages)} pages into {OUT}')


if __name__ == '__main__':
    main()
