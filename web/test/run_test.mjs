// Run web/test's module under Deno's WebGPU:
//   deno run --unstable-webgpu -A run_test.mjs <module.mjs> [graphs...]
// Passes only on a printed "RESULT: PASS" line: an exit code alone is not
// evidence (an Asyncify main can report one before it has run).
const [mod, ...graphs] = Deno.args;
const { default: factory } = await import(new URL(mod, `file://${Deno.cwd()}/`).href);
let last = '';
const m = await factory({
  print: (s) => { last = s; console.log(s); },
  printErr: (s) => console.error(s),
});
const rc = await m.ccall('web_main', 'number', ['string'], [graphs.join(' ')], { async: true });
const pass = rc === 0 && last === 'RESULT: PASS';
if (!pass) console.log(`run_test: web_main returned ${rc}, last line "${last}"`);
Deno.exit(pass ? 0 : 1);
