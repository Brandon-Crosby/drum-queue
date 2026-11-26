#!/usr/bin/env node
/* Compile with the same WASM Python used in the browser, never the host Python.
 * Checked-hash pyc files are ignored if their corresponding source changes.
 */
const fs = require('node:fs/promises');
const path = require('node:path');
const { createHash } = require('node:crypto');

async function main() {
  const [runtimeArg, outputArg] = process.argv.slice(2);
  if (!runtimeArg || !outputArg) throw Error('Usage: node tools/browser/prepare_bytecode.cjs RUNTIME OUTPUT');
  const runtime = path.resolve(runtimeArg), output = path.resolve(outputArg);
  const web = path.resolve(__dirname, '../../web');
  if (output === web || output.startsWith(web + path.sep))
    throw Error('Keep prepared runtime assets outside web/ and the source package');
  const { loadPyodide } = require(path.join(runtime, 'pyodide.js'));
  const py = await loadPyodide({ indexURL: runtime + path.sep });
  if (py.version !== '0.29.0') throw Error('Expected Pyodide 0.29.0');
  await py.loadPackage(['numpy', 'scipy', 'soundfile']);
  const result = await py.runPythonAsync(`
import compileall, py_compile, pathlib, zipfile, json
root = pathlib.Path('/lib/python3.13')
if not compileall.compile_dir(str(root), quiet=1, force=True,
                             invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH):
    raise RuntimeError('Could not compile the runtime sources')
files = sorted(root.rglob('*.pyc'))
with zipfile.ZipFile('/bytecode.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for source in files:
        # Hash-based cache entries are tied to the source, rather than VFS timestamps.
        if source.read_bytes()[4:8] != b'\\x03\\x00\\x00\\x00':
            continue
        info = zipfile.ZipInfo(str(source).lstrip('/'), (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, source.read_bytes())
with zipfile.ZipFile('/bytecode.zip') as archive:
    count = len(archive.namelist())
json.dumps(dict(runtime='Pyodide 0.29.0', files=count, invalidation='checked-hash'))
`);
  const bytes = py.FS.readFile('/bytecode.zip');
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const lock = await fs.readFile(path.join(runtime, 'pyodide-lock.json'));
  const report = { ...JSON.parse(result), bytes: bytes.byteLength, sha256,
    runtimeLockSha256: createHash('sha256').update(lock).digest('hex') };
  await fs.mkdir(output, { recursive: true });
  await fs.writeFile(path.join(output, 'bytecode.zip'), bytes);
  await fs.writeFile(path.join(output, 'manifest.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report));
}
main().catch(error => { console.error(String(error)); process.exitCode = 1; });
