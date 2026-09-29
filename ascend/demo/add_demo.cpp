// add_demo.cpp — minimal Ascend operator demo to validate MSPTI (LD_PRELOAD) tracing.
//
// This is a *clean* ACL program: no MSPTI API calls at all. Tracing is done
// entirely by the injected libmspti.so + libcolamspti.so (see build_and_run.sh).
// It periodically (once per second) runs the full aclnnAdd pipeline, driving
// exactly the paths that parcagpu's mspti.cpp subscribes to:
//
//   aclrtMemcpyAsync (H2D/D2H) -> MSPTI_CBID_RUNTIME_MEMCPY_ASYNC  + MEMCPY activity
//   aclnnAdd                   -> MSPTI_CBID_RUNTIME_LAUNCH        + KERNEL activity
//   aclrtSynchronizeStream     -> MSPTI_CBID_RUNTIME_STREAM_SYNCHRONIZED
//
// Usage:
//   ./add_demo          # run forever (Ctrl-C to stop)
//   ./add_demo 60       # run exactly 60 iterations (one per second)
//
// The device/context/stream init order mirrors Huawei's own MSPTI sample
// (samples/common/util_acl.h): the injected callbacks only become live after a
// real NPU call, so these first calls are what "wake up" MSPTI.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"

#define ACL_CHECK(expr)                                                       \
  do {                                                                        \
    aclError _e = (expr);                                                     \
    if (_e != ACL_SUCCESS) {                                                  \
      fprintf(stderr, "[add_demo] %s failed: %d (%s:%d)\n", #expr, (int)_e,  \
              __FILE__, __LINE__);                                            \
      return -1;                                                              \
    }                                                                         \
  } while (0)

int main(int argc, char **argv) {
  const int32_t deviceId = 0;
  const int64_t N = 1 << 20; // 1M floats (4 MB/tensor): large enough for a
                             // non-zero, observable KERNEL duration.
  const int periodMs = 1000; // one full op per second
  // iterations == 0 -> run until Ctrl-C, otherwise run exactly `iterations`.
  const int iterations = (argc > 1) ? std::atoi(argv[1]) : 0;

  // ---- 1. init device / context / stream (wakes up injected MSPTI) --------
  // 与华为 sample 的 Init() 顺序保持一致。
  ACL_CHECK(aclrtSetDevice(deviceId));
  aclrtContext context = nullptr;
  ACL_CHECK(aclrtCreateContext(&context, deviceId));
  ACL_CHECK(aclrtSetCurrentContext(context));
  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclInit(nullptr));

  // ---- 2. build 1-D tensors + device buffers (allocated once) ------------
  std::vector<int64_t> shape = {N};
  std::vector<int64_t> strides = {1};
  std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N, 0.0f);
  const size_t bytes = N * sizeof(float);

  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  ACL_CHECK(aclrtMalloc(&dA, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(&dB, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(&dC, bytes, ACL_MEM_MALLOC_HUGE_FIRST));

  aclTensor *tA =
      aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                      ACL_FORMAT_ND, shape.data(), shape.size(), dA);
  aclTensor *tB =
      aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                      ACL_FORMAT_ND, shape.data(), shape.size(), dB);
  aclTensor *tC =
      aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                      ACL_FORMAT_ND, shape.data(), shape.size(), dC);
  float alphaValue = 1.0f;
  aclScalar *alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  if (!tA || !tB || !tC || !alpha) {
    fprintf(stderr, "[add_demo] tensor/scalar creation failed\n");
    return -1;
  }

  // ---- 3. workspace (buffer reused; executor is per-iteration) ------------
  // aclnn two-phase API: aclOpExecutor is single-use — aclnnAdd consumes it, so
  // GetWorkspaceSize must be called fresh each iteration. The workspace buffer
  // (device memory) is stable for a fixed shape, so allocate it lazily once.
  uint64_t wsSize = 0;
  void *ws = nullptr;

  printf("[add_demo] running aclnnAdd every %d ms (iterations=%s)\n", periodMs,
         iterations == 0 ? "until Ctrl-C" : argv[1]);
  fflush(stdout);

  // ---- 4. periodic loop ----------------------------------------------------
  auto next = std::chrono::steady_clock::now();
  for (int i = 0; iterations == 0 || i < iterations; ++i) {
    // fresh executor per iteration (single-use)
    aclOpExecutor *executor = nullptr;
    ACL_CHECK(aclnnAddGetWorkspaceSize(tA, tB, alpha, tC, &wsSize, &executor));
    if (ws == nullptr && wsSize > 0) {
      ACL_CHECK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // async H2D -> MEMCPY_ASYNC cbid + MEMCPY activity
    ACL_CHECK(aclrtMemcpyAsync(dA, bytes, a.data(), bytes,
                               ACL_MEMCPY_HOST_TO_DEVICE, stream));
    ACL_CHECK(aclrtMemcpyAsync(dB, bytes, b.data(), bytes,
                               ACL_MEMCPY_HOST_TO_DEVICE, stream));

    // launch -> LAUNCH cbid + KERNEL activity
    ACL_CHECK(aclnnAdd(ws, wsSize, executor, stream));

    // sync -> STREAM_SYNCHRONIZED cbid
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // async D2H -> MEMCPY_ASYNC cbid + MEMCPY activity
    ACL_CHECK(aclrtMemcpyAsync(c.data(), bytes, dC, bytes,
                               ACL_MEMCPY_DEVICE_TO_HOST, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    printf("[add_demo] iter=%d c[0]=%.1f (expect 3.0)\n", i, c[0]);
    fflush(stdout);

    // sleep until the next 1s boundary (fixed-rate cadence)
    next += std::chrono::milliseconds(periodMs);
    std::this_thread::sleep_until(next);
  }

  // ---- 5. cleanup ---------------------------------------------------------
  aclDestroyTensor(tA);
  aclDestroyTensor(tB);
  aclDestroyTensor(tC);
  aclDestroyScalar(alpha);
  aclrtFree(dA);
  aclrtFree(dB);
  aclrtFree(dC);
  if (ws) {
    aclrtFree(ws);
  }
  ACL_CHECK(aclrtDestroyStream(stream));
  ACL_CHECK(aclrtDestroyContext(context));
  ACL_CHECK(aclrtResetDevice(deviceId));
  ACL_CHECK(aclFinalize());
  return 0;
}
