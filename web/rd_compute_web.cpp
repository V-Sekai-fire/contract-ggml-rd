// rdc::Device on WebGPU (web/rd_compute.h); the GPU objects live in RD,
// web/rd_webgpu.js.
#include "rd_compute.h"

#include <emscripten.h>

#include <cstdio>
#include <cstdlib>

namespace {

EM_ASYNC_JS(int, js_open, (), { return await RD.open(); });
EM_JS(int, js_errors, (), { return RD.errors.length; });
EM_JS(char *, js_take_error, (), {
	const e = RD.errors.splice(0).join('; ');
	return stringToNewUTF8(e);
});
EM_JS(char *, js_name, (), { return stringToNewUTF8(RD.name()); });
EM_JS(int, js_shader, (const uint8_t *src, size_t bytes), {
	const code = UTF8ToString(src, bytes);
	return RD.put(RD.device.createShaderModule({ code }));
});
EM_ASYNC_JS(int, js_pipeline, (int shader), { return await RD.pipeline(shader); });
EM_JS(int, js_storage, (double bytes), {
	const size = Math.max(4, Math.ceil(bytes / 4) * 4);
	return RD.put(RD.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }));
});
EM_JS(int, js_uniform, (double bytes, const void *data), {
	const size = Math.max(16, Math.ceil(bytes / 16) * 16);
	const b = RD.device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
	if (data) new Uint8Array(b.getMappedRange()).set(HEAPU8.subarray(data, data + bytes));
	b.unmap();
	return RD.put(b);
});
EM_ASYNC_JS(int, js_write, (int buf, double off, double bytes, const void *src), { return await RD.write(buf, off, bytes, src); });
EM_ASYNC_JS(int, js_read, (int buf, double off, double bytes, void *out), { return await RD.read(buf, off, bytes, out); });
EM_JS(int, js_copy, (int src, int dst, double bytes, double so, double dof), {
	if (bytes % 4 || so % 4 || dof % 4) return 0;
	const enc = RD.device.createCommandEncoder();
	enc.copyBufferToBuffer(RD.get(src), so, RD.get(dst), dof, bytes);
	RD.device.queue.submit([enc.finish()]);
	return 1;
});
EM_JS(int, js_clear, (int buf, double off, double bytes), {
	if (bytes % 4 || off % 4) return 0;
	const enc = RD.device.createCommandEncoder();
	enc.clearBuffer(RD.get(buf), off, bytes);
	RD.device.queue.submit([enc.finish()]);
	return 1;
});
EM_JS(int, js_bind_group, (int set, const int *binding, const int *rid, const double *off, const double *size, int n), {
	const entries = [];
	for (let i = 0; i < n; i++) {
		const r = HEAP32[(rid >> 2) + i];
		const o = HEAPF64[(off >> 3) + i], z = HEAPF64[(size >> 3) + i];
		const bi = HEAP32[(binding >> 2) + i];
		const resource = r ? { buffer: RD.get(r), offset: o } : { buffer: RD.dummies[bi] };
		if (r && z > 0) resource.size = z;
		entries.push({ binding: bi, resource });
	}
	if (set === 0) {
		// The layout has all five bindings; an unbound source gets the dummy.
		for (let b = 0; b < 5; b++) if (!entries.some((e) => e.binding === b)) entries.push({ binding: b, resource: { buffer: RD.dummies[b] } });
	}
	return RD.put(RD.device.createBindGroup({ layout: RD.layouts[set], entries }));
});
EM_JS(void, js_drop, (int i, int destroy), { RD.drop(i, destroy); });
EM_JS(void, js_list_begin, (), {
	RD.encoder = RD.device.createCommandEncoder();
	RD.pass = RD.encoder.beginComputePass();
});
EM_JS(void, js_bind_pipeline, (int p), { RD.pass.setPipeline(RD.get(p)); });
EM_JS(void, js_bind_set, (int s, int set), { RD.pass.setBindGroup(set, RD.get(s)); });
EM_JS(void, js_dispatch, (int x, int y, int z), { RD.pass.dispatchWorkgroups(x, y, z); });
EM_JS(void, js_list_copy, (int src, double so, int dst, double dof, double bytes), {
	RD.pass.end();
	RD.encoder.copyBufferToBuffer(RD.get(src), so, RD.get(dst), dof, bytes);
	RD.pass = RD.encoder.beginComputePass();
});
EM_JS(void, js_list_end, (), {
	RD.pass.end();
	RD.pass = null;
});
EM_JS(void, js_submit, (), {
	RD.device.queue.submit([RD.encoder.finish()]);
	RD.encoder = null;
});
EM_ASYNC_JS(void, js_sync, (), { await RD.device.queue.onSubmittedWorkDone(); });

} // namespace

namespace rdc {

int64_t host_usec() {
	return int64_t(emscripten_get_now() * 1000.0);
}

bool Device::fail(const std::string &why) {
	err_ = why;
	if (js_errors()) {
		char *e = js_take_error();
		err_ += std::string(": ") + e;
		std::free(e);
	}
	return false;
}

bool Device::open() {
	if (open_) {
		return true;
	}
	const int mb = js_open();
	if (mb <= 0) {
		return fail("open");
	}
	// ggml-rd's max_buffer_bytes(): keep every buffer bindable whole.
	const std::string v = std::to_string(mb);
	setenv("GGML_RD_MAX_BUFFER_MB", v.c_str(), 1);
	open_ = true;
	return true;
}

void Device::close() {
	open_ = false;
}

::RID Device::shader_from_spirv(const uint8_t *wgsl, size_t bytes) {
	return ::RID{ js_shader(wgsl, bytes) };
}

::RID Device::compute_pipeline(::RID shader) {
	::RID r{ js_pipeline(int(shader.index)) };
	if (r.index == 0) {
		fail("compute_pipeline");
	}
	return r;
}

::RID Device::storage_buffer_empty(size_t bytes, bool) {
	return ::RID{ js_storage(double(bytes)) }; // WebGPU zero-fills new buffers
}

::RID Device::uniform_buffer(size_t bytes, const void *data) {
	return ::RID{ js_uniform(double(bytes), data) };
}

bool Device::buffer_update(::RID buffer, size_t offset, size_t bytes, const void *data) {
	return js_write(int(buffer.index), double(offset), double(bytes), data) != 0 || fail("buffer_update");
}

bool Device::buffer_get_into(::RID buffer, size_t offset, size_t bytes, void *out) {
	return js_read(int(buffer.index), double(offset), double(bytes), out) != 0 || fail("buffer_get_into");
}

bool Device::buffer_copy(::RID src, ::RID dst, size_t bytes, size_t src_offset, size_t dst_offset) {
	return js_copy(int(src.index), int(dst.index), double(bytes), double(src_offset), double(dst_offset)) != 0 ||
			fail("buffer_copy: unaligned");
}

bool Device::buffer_clear(::RID buffer, size_t offset, size_t bytes) {
	return js_clear(int(buffer.index), double(offset), double(bytes)) != 0 || fail("buffer_clear: unaligned");
}

::RID Device::uniform_set(::RID, const std::vector<Binding> &bindings, uint32_t set) {
	std::vector<int> b, r;
	std::vector<double> o, z;
	for (const Binding &x : bindings) {
		b.push_back(int(x.binding));
		r.push_back(int(x.rid.index));
		o.push_back(double(x.offset));
		z.push_back(double(x.size));
	}
	return ::RID{ js_bind_group(int(set), b.data(), r.data(), o.data(), z.data(), int(b.size())) };
}

void Device::free_rid(::RID rid) {
	js_drop(int(rid.index), 1);
}

void Device::forget(::RID rid) {
	js_drop(int(rid.index), 0);
}

std::string Device::device_name() {
	char *n = js_name();
	std::string s(n);
	std::free(n);
	return s;
}

bool Device::list_begin() {
	js_list_begin();
	return true;
}

void Device::bind_pipeline(::RID pipeline) {
	js_bind_pipeline(int(pipeline.index));
}

void Device::bind_uniform_set(::RID uniform_set, uint32_t set) {
	js_bind_set(int(uniform_set.index), int(set));
}

void Device::dispatch(uint32_t x, uint32_t y, uint32_t z) {
	js_dispatch(int(x), int(y), int(z));
}

void Device::list_copy(::RID src, uint64_t src_offset, ::RID dst, uint64_t dst_offset, uint64_t bytes) {
	js_list_copy(int(src.index), double(src_offset), int(dst.index), double(dst_offset), double(bytes));
}

void Device::list_end() {
	js_list_end();
}

void Device::submit() {
	js_submit();
}

void Device::sync() {
	js_sync();
	if (js_errors()) {
		fail("webgpu");
		std::fprintf(stderr, "ggml-rd web: %s\n", err_.c_str());
	}
}

} // namespace rdc
