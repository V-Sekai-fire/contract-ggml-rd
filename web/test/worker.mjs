// web/test's module in a Web Worker: run web_main and post every printed line.
import factory from './ggml_rd_web_test.mjs';

self.onmessage = async (e) => {
  let last = '';
  const post = (line) => { last = line; self.postMessage({ line }); };
  try {
    const m = await factory({ print: post, printErr: post });
    const rc = await m.ccall('web_main', 'number', ['string'], [e.data.graphs ?? ''], { async: true });
    self.postMessage({ done: true, rc, pass: rc === 0 && last === 'RESULT: PASS' });
  } catch (err) {
    post(`worker: ${err?.stack ?? err}`);
    self.postMessage({ done: true, rc: -1, pass: false });
  }
};
