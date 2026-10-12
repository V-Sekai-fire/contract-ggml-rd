// Force-included before the slangc cpp emits in the web build: slang's prelude
// does not know emscripten, so it is described here as the Unix it behaves as.
#pragma once
#define SLANG_PLATFORM
#define SLANG_WINRT 0
#define SLANG_XBOXONE 0
#define SLANG_WIN64 0
#define SLANG_X360 0
#define SLANG_WIN32 0
#define SLANG_ANDROID 0
#define SLANG_LINUX 1
#define SLANG_IOS 0
#define SLANG_OSX 0
#define SLANG_PS3 0
#define SLANG_PS4 0
#define SLANG_PSP2 0
#define SLANG_WIIU 0
// wasm32 has no _Float16, but float.h would announce one (FLT16_MIN) once the
// prelude asks for it. Including float.h first, unasked, keeps the prelude on
// its portable `struct half`.
#include <float.h>
