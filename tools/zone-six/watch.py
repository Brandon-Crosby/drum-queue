#!/usr/bin/env python3
"""Called from trusted main by the hourly workflow. Never execute website HTML."""
import json, os, re, subprocess, tempfile, urllib.request
from pathlib import Path
BASE='https://api.github.com/repos/Brandon-Crosby/drum-queue'
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ZoneSix-desktop-sync','Cache-Control':'no-cache'}),timeout=20) as r:
        data=r.read(1024*1024+1)
        if len(data)>1024*1024:raise ValueError('Response exceeds limit')
        return json.loads(data)
def deployed_revision():
    # A protected site may return a Cloudflare Access page. Use the trusted
    # Pages integration's successful check for current main, never scraped HTML.
    sha=get(BASE+'/commits/main')['sha']
    assert re.fullmatch('[a-f0-9]{40}',sha), 'Invalid main revision'
    try:
        marker=get('https://drumsplitter.com/_zone-six/release.json')
        if marker.get('schemaVersion')==1 and marker.get('repository')=='Brandon-Crosby/drum-queue' and marker.get('sourceCommit')==sha:
            return sha
    except (ValueError, OSError):
        pass
    checks=get(BASE+'/commits/'+sha+'/check-runs?per_page=100')['check_runs']
    assert any(c.get('name')=='Cloudflare Pages' and c.get('app',{}).get('slug')=='cloudflare-workers-and-pages'
        and c.get('head_sha')==sha and c.get('status')=='completed' and c.get('conclusion')=='success' for c in checks), 'Waiting for a successful Cloudflare Pages deployment of main'
    return sha

def main():
    sha=deployed_revision()
    tag='zone-six-source-'+sha
    releases=get(BASE+'/releases?per_page=30')
    if any(r['tag_name']==tag and not r['draft'] for r in releases):
        print('Desktop source already published');return
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)/'repo';out=Path(td)/'out'
        subprocess.run(['git','clone','--quiet','--depth','1','https://github.com/Brandon-Crosby/drum-queue.git',str(root)],check=True)
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==sha, 'Main changed while building; retry'
        subprocess.run(['python3',str(root/'tools/zone-six/build.py'),'--revision',sha,'--output',str(out)],check=True)
        subprocess.run(['python3',str(root/'tools/zone-six/test_contract.py'),'--source-zip',str(out/'drumsplitter-source.zip')],check=True)
        # Recheck deployment before publication. A failed build leaves the previous release intact.
        assert deployed_revision()==sha, 'Deployment changed while testing'
        subprocess.run(['gh','release','create',tag,str(out/'drumsplitter-source.zip'),str(out/'drumsplitter-source.json'),'--repo','Brandon-Crosby/drum-queue','--target',sha,'--prerelease','--latest=false','--title','Zone Six desktop source '+sha[:12],'--notes','Automatically built from the deployed website commit. Source/interface only; no model or runtime. Zone Six validates against the installed Mac runtime before activation.'],check=True)
if __name__=='__main__':main()
