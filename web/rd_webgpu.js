// The WebGPU side of web/rd_compute_web.cpp (--pre-js): an object table
// whose integer slots are the RIDs ggml-rd holds. Slot 0 is the null RID.
var RD = {
  objs: [null],
  free: [],
  adapter: null,
  device: null,
  layouts: null,
  pipelineLayout: null,
  dummy: null,
  encoder: null,
  pass: null,
  pending: [],
  errors: [],
  put(o) {
    const i = RD.free.length ? RD.free.pop() : RD.objs.length;
    RD.objs[i] = o;
    return i;
  },
  get(i) { return RD.objs[Number(i)]; },
  drop(i, destroy) {
    i = Number(i);
    const o = RD.objs[i];
    if (!o) return;
    if (destroy && o.destroy) o.destroy();
    RD.objs[i] = null;
    RD.free.push(i);
  },
  async open() {
    if (RD.device) return RD.limitMB();
    if (typeof navigator === 'undefined' || !navigator.gpu) { RD.errors.push('no navigator.gpu'); return 0; }
    RD.adapter = await navigator.gpu.requestAdapter();
    if (!RD.adapter) { RD.errors.push('no WebGPU adapter'); return 0; }
    // Default limits on purpose: what every WebGPU device offers.
    RD.device = await RD.adapter.requestDevice();
    RD.device.addEventListener?.('uncapturederror', (e) => { RD.errors.push(e.error.message); console.error('webgpu:', e.error.message); });
    const S = GPUShaderStage.COMPUTE;
    RD.layouts = [
      RD.device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: S, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: S, buffer: { type: 'storage' } },
        { binding: 2, visibility: S, buffer: { type: 'storage' } },
        { binding: 3, visibility: S, buffer: { type: 'storage' } },
        { binding: 4, visibility: S, buffer: { type: 'storage' } },
      ] }),
      RD.device.createBindGroupLayout({ entries: [{ binding: 0, visibility: S, buffer: { type: 'uniform' } }] }),
    ];
    RD.pipelineLayout = RD.device.createPipelineLayout({ bindGroupLayouts: RD.layouts });
    RD.dummy = RD.device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE });
    return RD.limitMB();
  },
  limitMB() {
    const l = RD.device.limits;
    return Math.floor(Math.min(l.maxStorageBufferBindingSize, l.maxBufferSize) / 1048576);
  },
  name() {
    const i = RD.adapter?.info ?? {};
    return `WebGPU ${i.vendor ?? ''} ${i.architecture ?? ''} ${i.device ?? ''}`.trim();
  },
  async pipeline(shader) {
    try {
      return RD.put(await RD.device.createComputePipelineAsync({ layout: RD.pipelineLayout, compute: { module: RD.get(shader), entryPoint: 'main' } }));
    } catch (e) {
      RD.errors.push(String(e.message ?? e));
      return 0;
    }
  },
  async read(buf, off, bytes, out) {
    const lo = off - (off % 4), hi = Math.ceil((off + bytes) / 4) * 4;
    const st = RD.device.createBuffer({ size: hi - lo, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = RD.device.createCommandEncoder();
    enc.copyBufferToBuffer(RD.get(buf), lo, st, 0, hi - lo);
    RD.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    HEAPU8.set(new Uint8Array(st.getMappedRange(), off - lo, bytes), out);
    st.unmap();
    st.destroy();
    return 1;
  },
  async write(buf, off, bytes, src) {
    const b = RD.get(buf);
    const lo = off - (off % 4), hi = Math.ceil((off + bytes) / 4) * 4;
    if (lo === off && hi === off + bytes) {
      RD.device.queue.writeBuffer(b, off, HEAPU8, src, bytes);
      return 1;
    }
    // writeBuffer moves whole words: read the edge words back and merge.
    const words = new Uint8Array(hi - lo);
    const ptr = _malloc(hi - lo);
    await RD.read(buf, lo, hi - lo, ptr);
    words.set(HEAPU8.subarray(ptr, ptr + hi - lo));
    _free(ptr);
    words.set(HEAPU8.subarray(src, src + bytes), off - lo);
    RD.device.queue.writeBuffer(b, lo, words);
    return 1;
  },
};
