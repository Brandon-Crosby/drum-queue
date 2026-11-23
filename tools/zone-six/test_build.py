#!/usr/bin/env python3
import importlib.util,json,os,shutil,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('builder',ROOT/'tools/zone-six/build.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
class BuildTests(unittest.TestCase):
    def test_website_marker(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);shutil.copytree(ROOT/'web',root/'web',ignore=shutil.ignore_patterns('dist','__pycache__'))
            subprocess.run(['python3',str(root/'web/build_site.py')],check=True,env={**os.environ,'CF_PAGES_COMMIT_SHA':'a'*40})
            m=json.loads((root/'web/dist/_zone-six/release.json').read_text());self.assertEqual(m['sourceCommit'],'a'*40)
            self.assertEqual(set(m),{'schemaVersion','repository','sourceCommit'})
    def test_dependency_change_blocks_build(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'source';root.mkdir()
            for name in ['engine.py','oneshots.py','requirements.txt','kitmatch','web']:
                p=ROOT/name
                if p.is_dir():shutil.copytree(p,root/name,ignore=shutil.ignore_patterns('dist','__pycache__'))
                else:shutil.copy2(p,root/name)
            (root/'requirements.txt').write_text('torch==999\n')
            with self.assertRaisesRegex(AssertionError,'Dependencies changed'):b.build(root,Path(td)/'out','a'*40)
    def test_invalid_revision(self):
        with self.assertRaises(AssertionError):b.build(ROOT,Path('/tmp/not-created'),'../../main')
if __name__=='__main__':unittest.main()
