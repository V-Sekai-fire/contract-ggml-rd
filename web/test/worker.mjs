// web/test's module in a Web Worker: load the references, run web_main and
// post every printed line.
import factory from './ggml_rd_web_test.mjs';
import { loadRefs } from './refs.mjs';

self.onmessage = async (e) => {
  let last = '';
  const post = (line) => { last = line; self.postMessage({ line }); };
  try {
    const graphs = (e.data.graphs ?? '').split(/[ +]/).filter(Boolean);
    const m = await factory({ print: post, printErr: post });
    const r = await loadRefs(m, new URL('./refs/', self.location.href), graphs);
    post(`refs: ${r.outputs} outputs, ${(r.bytes / 1048576).toFixed(1)} MiB`);
    const rc = await m.ccall('web_main', 'number', ['string'], [graphs.join(' ')], { async: true });
    self.postMessage({ done: true, rc, pass: rc === 0 && last === 'RESULT: PASS' });
  } catch (err) {
    post(`worker: ${err?.stack ?? err}`);
    self.postMessage({ done: true, rc: -1, pass: false });
  }
};
