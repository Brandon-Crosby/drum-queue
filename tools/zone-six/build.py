#!/usr/bin/env python3
"""Build a source-only Mac component from the exact deployed website commit.
No Python runtime, model weights, credentials, uploads or training data enter this ZIP.
"""
import argparse, ast, hashlib, json, os, re, shutil, subprocess, tempfile, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = 'Brandon-Crosby/drum-queue'
MSST = '1d4b3c431b37de696b3795ed2018493aacba946f'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dependency_fingerprint(root):
    files = ['requirements.txt', 'web/backend/requirements.txt']
    return {name: digest(root / name) for name in files}

def contract(root):
    engine = ast.parse((root/'engine.py').read_text(), feature_version=(3,11))
    classes = {n.name:n for n in engine.body if isinstance(n,ast.ClassDef)}
    init = next(n for n in classes['Models'].body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assert {'helyx','device','soundid_ref'} <= {n.arg for n in init.args.args}, 'Models API changed'
    service = ast.parse((root/'web/backend/service.py').read_text(),feature_version=(3,11))
    functions = {n.name:n for n in service.body if isinstance(n,ast.FunctionDef)}
    assert {'settings','cancel'} <= {n.arg for n in functions['create_app'].args.args}, 'Local API changed'
    assert {'models','mix_model','max_seconds'} <= {n.arg for n in functions['run_job'].args.args}, 'Job API changed'
    html=(root/'web/dist/index.html').read_text()
    assert 'window.SITE={local:true,' in html and 'adsbygoogle.js' not in html and 'turnstile/v0/api.js' not in html, 'Local UI build failed'
    for path in root.rglob('*.py'):
        ast.parse(path.read_text(),filename=str(path),feature_version=(3,11))

def build(source, output, revision):
    assert re.fullmatch('[a-f0-9]{40}',revision), 'Invalid revision'
    for name in ['engine.py','oneshots.py','kitmatch','web']:
        p=source/name
        assert not p.is_symlink() and not any(x.is_symlink() for x in p.rglob('*')), 'Symlink in upstream source'
    baseline=json.loads((ROOT/'tools/zone-six/compatibility.json').read_text())
    assert dependency_fingerprint(source)==baseline['dependencies'], 'Dependencies changed: runtime rebuild required'
    modal=(source/'web/backend/modal_app.py').read_text()
    assert re.search(r"MSST_COMMIT\s*=\s*['\"]"+MSST+r"['\"]",modal), 'MSST version changed'
    assert re.search(r"HELYX_RUN\s*=.*['\"]runs/helyx-4['\"]",modal), 'Helyx model changed'
    with tempfile.TemporaryDirectory() as td:
        stage=Path(td)/'source'; stage.mkdir()
        for name in ['engine.py','oneshots.py','kitmatch','web']:
            p=source/name
            if p.is_dir():shutil.copytree(p,stage/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','dist','.local_data'))
            else:shutil.copy2(p,stage/name)
        # Patch a disposable copy, never the live website source.
        subprocess.run(['git','apply',str(ROOT/'tools/zone-six/macos.patch')],check=True,cwd=stage)
        subprocess.run(['python3',str(stage/'web/build_site.py')],check=True,env={**os.environ,'LOCAL_MODE':'true','ADSENSE_CLIENT':'','TURNSTILE_SITEKEY':'','CF_PAGES_COMMIT_SHA':revision})
        contract(stage)
        for js in (stage/'web/dist').rglob('*.js'):
            subprocess.run(['node','--check',str(js)],check=True)
        # Only the loopback adapter's code and rendered public interface are distributed.
        shutil.rmtree(stage/'web/site')
        for p in list((stage/'web').iterdir()):
            if p.name not in ['backend','dist']:
                if p.is_dir():shutil.rmtree(p)
                else:p.unlink()
        for p in (stage/'web/backend').iterdir():
            if p.name!='service.py':
                if p.is_dir():shutil.rmtree(p)
                else:p.unlink()
        for p in (stage/'kitmatch').rglob('*'):
            if p.is_file() and p.suffix!='.py':p.unlink()
        for p in stage.rglob('*'):
            assert not p.is_symlink(), 'Symlink in source'
            if p.is_file():assert p.stat().st_size<=10*1024*1024, 'Unexpected large source file'
        output.mkdir(parents=True,exist_ok=True)
        package=output/'drumsplitter-source.zip'
        with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(stage.rglob('*')):
                if p.is_file():z.writestr('source/'+p.relative_to(stage).as_posix(),p.read_bytes())
        assert package.stat().st_size<=20*1024*1024, 'Source package exceeds limit'
        manifest={'schemaVersion':1,'repository':REPOSITORY,'sourceCommit':revision,'minimumHub':'0.6.0','runtimeProtocol':1,'backendVersion':'0.1.0','model':'helyx-4','msstCommit':MSST,'dependencies':baseline['dependencies'],'filename':package.name,'bytes':package.stat().st_size,'sha256':digest(package)}
        (output/'drumsplitter-source.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True);p.add_argument('--revision',required=True)
    a=p.parse_args();print(json.dumps(build(a.source,a.output,a.revision)))
