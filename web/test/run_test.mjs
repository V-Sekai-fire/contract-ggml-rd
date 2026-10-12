// Run web/test's module under Deno's WebGPU:
//   deno run --unstable-webgpu -A run_test.mjs <module.mjs> <refs dir> [graphs...]
// Passes only on a printed "RESULT: PASS" line: an exit code alone is not
// evidence (an Asyncify main can report one before it has run).
import { loadRefs } from './refs.mjs';

const [mod, refs, ...graphs] = Deno.args;
const cwd = `file://${Deno.cwd()}/`;
const { default: factory } = await import(new URL(mod, cwd).href);
let last = '';
const env = Object.fromEntries(Object.entries(Deno.env.toObject()).filter(([k]) => k.startsWith('GGML_RD_')));
const m = await factory({
  print: (s) => { last = s; console.log(s); },
  printErr: (s) => console.error(s),
  preRun: [(mod) => Object.assign(mod.ENV, env)], // GGML_RD_* switches reach the module
});
const r = await loadRefs(m, new URL(refs.endsWith('/') ? refs : refs + '/', cwd), graphs);
console.log(`refs: ${r.outputs} outputs, ${(r.bytes / 1048576).toFixed(1)} MiB`);
const rc = await m.ccall('web_main', 'number', ['string'], [graphs.join(' ')], { async: true });
const pass = rc === 0 && last === 'RESULT: PASS';
if (!pass) console.log(`run_test: web_main returned ${rc}, last line "${last}"`);
Deno.exit(pass ? 0 : 1);
