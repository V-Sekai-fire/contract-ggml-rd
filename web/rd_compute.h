// rd_compute on WebGPU: the same rdc::Device calls ggml-rd makes on Godot's
// RenderingDevice (guest-runtime/guest/rd_compute.h), made on the browser's
// GPUDevice. Shaders are WGSL text (slangc -target wgsl of the same Slang
// kernels), passed where the guest passes SPIR-V. Every method that waits for
// the GPU (sync, buffer_get_into) suspends through Asyncify, so it must run off
// the page's main thread, in a worker.
#pragma once

#include <api.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rd_enums.h"

namespace rdc {

// A buffer at a binding slot; size 0 binds from offset to the end.
struct Binding {
	uint32_t binding;
	int type;
	::RID rid;
	uint64_t offset = 0;
	uint64_t size = 0;
};

int64_t host_usec();

class Device {
public:
	// Request an adapter and a device with the default limits, so the build
	// runs on any WebGPU implementation; GGML_RD_MAX_BUFFER_MB is set to the
	// device's storage binding limit so ggml splits buffers under it.
	bool open();
	bool ok() const { return open_; }
	void close();

	::RID shader_from_spirv(const uint8_t *wgsl, size_t bytes);
	::RID compute_pipeline(::RID shader);
	::RID storage_buffer_empty(size_t bytes, bool clear = true);
	::RID uniform_buffer(size_t bytes, const void *data = nullptr);
	bool buffer_update(::RID buffer, size_t offset, size_t bytes, const void *data);
	bool buffer_get_into(::RID buffer, size_t offset, size_t bytes, void *out);
	bool buffer_copy(::RID src, ::RID dst, size_t bytes, size_t src_offset = 0, size_t dst_offset = 0);
	bool buffer_clear(::RID buffer, size_t offset, size_t bytes);
	::RID uniform_set(::RID shader, const std::vector<Binding> &bindings, uint32_t set = 0);
	void free_rid(::RID rid);
	void forget(::RID rid);
	std::string device_name();
	void capture_timestamp(const std::string &) {}
	int64_t timestamps_count() { return 0; }
	int64_t timestamp_gpu_ns(int64_t) { return -1; }

	bool list_begin();
	void bind_pipeline(::RID pipeline);
	void bind_uniform_set(::RID uniform_set, uint32_t set = 0);
	void dispatch(uint32_t groups_x, uint32_t groups_y = 1, uint32_t groups_z = 1);
	// WebGPU orders the dispatches of one compute pass by itself.
	void barrier() {}
	// Inside a list: end the compute pass, copy, open the next pass (the
	// pipeline and sets must be bound again).
	void list_copy(::RID src, uint64_t src_offset, ::RID dst, uint64_t dst_offset, uint64_t bytes);
	void list_end();
	void submit();
	void sync();

	const std::string &error() const { return err_; }

private:
	bool fail(const std::string &why);
	bool open_ = false;
	std::string err_;
};

} // namespace rdc
