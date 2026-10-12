// tests/webgpu: the ggml-rd kernels' WGSL emits on a WebGPU device vs ggml-cpu.
//
//   ggml_rd_l2 --export=<cases>                       (tests/ggml_rd_kernels)
//   slangc <k>.slang -target wgsl -> <wgsl>/<k>.wgsl  (kernels/ggml/slang)
//   deno run --unstable-webgpu -A run.mjs <cases> <wgsl> <kernels.txt>
//
// Each case's leaves go into one buffer, bound at every storage binding the
// kernel declares (word offsets are relative to it, as in the host harness);
// its dispatches run in order in one compute pass, and the output is compared
// with the ggml-cpu reference at the case's NMSE bound (test-backend-ops'
// rule). Control: every case again with its last dispatch left out must FAIL,
// unless its output equals its input there (the case's value_blind, or an
// identity such as LEAKY_RELU with slope 1 in place: counted as NOOP).
// The last line is "RESULT: PASS" or "RESULT: FAIL".
const [casesDir, wgslDir, kernelsTxt] = Deno.args;
const names = (await Deno.readTextFile(kernelsTxt)).split('\n').map((l) => l.trim()).filter((l) => l && !l.startsWith('#'));

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
});
device.addEventListener?.('uncapturederror', (e) => console.log('uncaptured', e.error.message));

const pipelines = new Map();
async function pipeline(id) {
  if (pipelines.has(id)) return pipelines.get(id);
  const code = await Deno.readTextFile(`${wgslDir}/${names[id]}.wgsl`);
  const binds = [...code.matchAll(/@binding\((\d+)\)\s*@group\((\d+)\)\s*var(?:<([^>]+)>)?\s+(\w+)/g)]
    .map((m) => ({ binding: +m[1], group: +m[2], kind: m[3] ?? '', name: m[4] }));
  device.pushErrorScope('validation');
  const p = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  const err = await device.popErrorScope();
  const out = { p, binds, err: err?.message };
  pipelines.set(id, out);
  return out;
}

const pad4 = (n) => (n + 3) & ~3;
function buffer(bytes, usage, data) {
  const b = device.createBuffer({ size: Math.max(16, pad4(bytes)), usage, mappedAtCreation: !!data });
  if (data) { new Uint8Array(b.getMappedRange()).set(data); b.unmap(); }
  return b;
}

function readOut(mem, out) {
  const dv = new DataView(mem.buffer, mem.byteOffset, mem.byteLength);
  const v = [];
  const [n0, n1, n2, n3] = out.ne, [b0, b1, b2, b3] = out.nb;
  for (let i3 = 0; i3 < n3; i3++) for (let i2 = 0; i2 < n2; i2++) for (let i1 = 0; i1 < n1; i1++) for (let i0 = 0; i0 < n0; i0++) {
    const o = out.off + i0 * b0 + i1 * b1 + i2 * b2 + i3 * b3;
    if (out.type === 'f32') v.push(dv.getFloat32(o, true));
    else if (out.type === 'i32') v.push(dv.getInt32(o, true));
    else if (out.type === 'bf16') { const t = new DataView(new ArrayBuffer(4)); t.setUint16(0, dv.getUint16(o, true)); v.push(t.getFloat32(0)); }
    else if (out.type === 'f16') v.push(f16(dv.getUint16(o, true)));
    else v.push(NaN);
  }
  return v;
}
function f16(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}
const big = (f) => !Number.isFinite(f) || Math.abs(f) >= 3.4028234663852886e38;
function compare(got, ref) {
  let ab = 0, a0 = 0, finite = true, exact = 0;
  for (let i = 0; i < ref.length; i++) {
    const a = got[i], b = ref[i];
    if (Object.is(a, b) || a === b) exact++;
    if (Number.isNaN(a) || Number.isNaN(b)) { finite = finite && Number.isNaN(a) && Number.isNaN(b); continue; }
    if (big(a) || big(b)) { if (!(big(a) && big(b) && Math.sign(a) === Math.sign(b))) finite = false; continue; }
    ab += (a - b) ** 2; a0 += a * a;
  }
  return { nmse: a0 > 0 ? ab / a0 : ab, finite, exact };
}

async function runCase(c, mem, skipLast) {
  const memBuf = buffer(mem.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mem);
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  const ds = skipLast ? c.dispatches.slice(0, -1) : c.dispatches;
  device.pushErrorScope('validation');
  for (const d of ds) {
    const { p, binds, err } = await pipeline(d.kernel);
    if (err) throw new Error(`${names[d.kernel]}: ${err}`);
    const params = buffer(4 * d.words.length, GPUBufferUsage.STORAGE, new Uint8Array(new Uint32Array(d.words).buffer));
    const slot = buffer(16, GPUBufferUsage.UNIFORM, new Uint8Array(16));
    for (const g of [...new Set(binds.map((b) => b.group))]) {
      const entries = binds.filter((b) => b.group === g).map((b) => ({
        binding: b.binding,
        resource: { buffer: b.kind.startsWith('uniform') ? slot : b.name.startsWith('params') ? params : memBuf },
      }));
      pass.setBindGroup(g, device.createBindGroup({ layout: p.getBindGroupLayout(g), entries }));
    }
    pass.setPipeline(p);
    pass.dispatchWorkgroups(...d.groups);
  }
  pass.end();
  const read = device.createBuffer({ size: memBuf.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  enc.copyBufferToBuffer(memBuf, 0, read, 0, memBuf.size);
  device.queue.submit([enc.finish()]);
  const verr = await device.popErrorScope();
  if (verr) throw new Error(verr.message);
  await read.mapAsync(GPUMapMode.READ);
  const out = new Uint8Array(read.getMappedRange().slice(0));
  read.unmap(); read.destroy(); memBuf.destroy();
  return readOut(out, c.out);
}

const files = [...Deno.readDirSync(casesDir)].filter((e) => e.name.endsWith('.json')).map((e) => e.name)
  .sort((a, b) => +a.match(/\d+/)[0] - +b.match(/\d+/)[0]);
console.log(`ggml-rd WebGPU: ${files.length} cases, ${names.length} kernels, adapter ${adapter.info?.vendor ?? ''} ${adapter.info?.architecture ?? ''}`);
let passed = 0, failed = 0, detected = 0, missed = 0, noop = 0;
const used = new Set();
for (const f of files) {
  const c = JSON.parse(await Deno.readTextFile(`${casesDir}/${f}`));
  const stem = `${casesDir}/${f.replace('.json', '')}`;
  const mem = await Deno.readFile(`${stem}.mem`);
  const ref = new Float32Array((await Deno.readFile(`${stem}.ref`)).buffer);
  let line = '', ok = false;
  try {
    if (c.error) throw new Error(c.error);
    c.dispatches.forEach((d) => used.add(names[d.kernel]));
    const r = compare(await runCase(c, mem, false), ref);
    ok = r.finite && r.nmse <= c.max_nmse;
    line = `nmse=${r.nmse.toExponential(3)} (max ${c.max_nmse}) exact=${r.exact}/${ref.length} kernels=${[...new Set(c.dispatches.map((d) => names[d.kernel]))].join('+')}`;
    const k = compare(await runCase(c, mem, true), ref);
    const caught = !(k.finite && k.nmse <= c.max_nmse);
    const identity = compare(readOut(mem, c.out), ref);
    const blind = c.value_blind || (identity.finite && identity.nmse <= c.max_nmse);
    if (caught) detected++;
    else if (blind) { noop++; line += ' control NOOP'; }
    else { missed++; line += ' CONTROL-MISSED'; }
  } catch (e) { line = `error: ${e.message.split('\n')[0].slice(0, 200)}`; }
  console.log(`${ok ? 'OK  ' : 'FAIL'} ${c.name.padEnd(44)} ${line}`);
  ok ? passed++ : failed++;
}
const all = new Set(names);
console.log(`summary: ${passed}/${files.length} cases OK; control (last dispatch left out) caught ${detected}, missed ${missed}, no-op ${noop}`);
console.log(`kernels exercised: ${used.size}/${all.size}; not exercised: ${[...all].filter((n) => !used.has(n)).join(' ')}`);
const pass = failed === 0 && passed > 0 && missed === 0;
console.log(`RESULT: ${pass ? 'PASS' : 'FAIL'}`);
Deno.exit(pass ? 0 : 1);
