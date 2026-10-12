// Hand web/refs' references to the test module (web_put_ref), for the nets
// asked for (all of them when none are). Shared by worker.mjs and run_test.mjs.
export async function loadRefs(m, base, graphs) {
  const list = await (await fetch(new URL('refs.json', base))).json();
  const want = graphs.length ? list.filter((r) => graphs.includes(r.graph)) : list;
  let bytes = 0;
  for (const r of want) {
    const res = await fetch(new URL(`${r.graph}/${r.arm}/${r.output}.f32`, base));
    if (!res.ok) throw new Error(`refs: ${r.graph}/${r.arm}/${r.output}: HTTP ${res.status}`);
    const f = new Float32Array(await res.arrayBuffer());
    if (f.length !== r.n) throw new Error(`refs: ${r.graph}/${r.arm}/${r.output}: ${f.length} floats, expected ${r.n}`);
    const p = m._malloc(f.byteLength);
    m.HEAPF32.set(f, p >> 2);
    m.ccall('web_put_ref', null, ['string', 'number', 'number'], [`${r.graph}/${r.arm}/${r.output}`, p, f.length]);
    m._free(p);
    bytes += f.byteLength;
  }
  return { outputs: want.length, bytes };
}
