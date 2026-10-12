// web/test's module in a Web Worker: load the references, run web_main and
// post every printed line. A private Hugging Face Space's signature
// (?__sign=..., from the page) rides on every request: the module, its wasm
// and the references.
const sign = new URLSearchParams(self.location.search).get('__sign');
const q = sign ? `?__sign=${encodeURIComponent(sign)}` : '';
const here = (p) => new URL(p, self.location.href).href.split('?')[0] + q;

self.onmessage = async (e) => {
  let last = '';
  const post = (line) => { last = line; self.postMessage({ line }); };
  try {
    const { default: factory } = await import(here('./ggml_rd_web_test.mjs'));
    const { loadRefs } = await import(here('./refs.mjs'));
    const graphs = (e.data.graphs ?? '').split(/[ +]/).filter(Boolean);
    const m = await factory({ print: post, printErr: post, locateFile: (p) => here(p) });
    const r = await loadRefs(m, new URL('./refs/', self.location.href), graphs, q);
    post(`refs: ${r.outputs} outputs, ${(r.bytes / 1048576).toFixed(1)} MiB`);
    const rc = await m.ccall('web_main', 'number', ['string'], [graphs.join(' ')], { async: true });
    self.postMessage({ done: true, rc, pass: rc === 0 && last === 'RESULT: PASS' });
  } catch (err) {
    post(`worker: ${err?.stack ?? err}`);
    self.postMessage({ done: true, rc: -1, pass: false });
  }
};
