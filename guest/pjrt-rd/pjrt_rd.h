// pjrt-rd: a PJRT C API plugin whose device is ggml-rd (RenderingDevice through
// rd_compute). It is linked into the guest, so GetPjrtApi() is a function call
// and not a dlopen. One PJRT_Buffer is one ggml-rd buffer. Compile takes the
// module that program.h writes (ggml ops kept as custom_calls); Execute rebuilds
// the ggml graph on the argument buffers and runs it on ggml-rd.
#pragma once

#include "ggml-backend.h"
#include "pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif
const PJRT_Api *GetPjrtApi(void);
#ifdef __cplusplus
}
#endif

namespace pjrt_rd {

// The ggml-rd backend the client's one device runs on; null detaches. The
// plugin never frees it.
void attach(ggml_backend_t backend);

// The text of the last error the plugin returned.
const char *last_error(void);

} // namespace pjrt_rd
