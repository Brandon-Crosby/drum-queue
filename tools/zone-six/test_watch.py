#!/usr/bin/env python3
import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('watcher',Path(__file__).with_name('watch.py'));w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
SHA='a'*40
class DeploymentTests(unittest.TestCase):
 def test_protected_site_requires_real_pages_success(self):
  check={'name':'Cloudflare Pages','app':{'slug':'cloudflare-workers-and-pages'},'head_sha':SHA,'status':'completed','conclusion':'success'}
  original=w.get
  def get(url):
   if url.startswith('https://drumsplitter.com/'):raise ValueError('Access login')
   return {'sha':SHA} if url.endswith('/main') else {'check_runs':[check]}
  try:
   w.get=get;self.assertEqual(w.deployed_revision(),SHA)
   for key,value in [('head_sha','b'*40),('status','in_progress'),('conclusion','failure'),('app',{'slug':'github-actions'})]:
    old=check[key];check[key]=value
    with self.assertRaises(AssertionError):w.deployed_revision()
    check[key]=old
  finally:w.get=original
if __name__=='__main__':unittest.main()
