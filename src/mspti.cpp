#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

// USDT probes — must come before any header that might include <sys/sdt.h>,
// so that _SDT_HAS_SEMAPHORES is defined first.
#include "probes.h"

#include "mspti.h"

#include "Utility/Singleton.h"
#include "correlation_filter.h"

#define DEBUG_PRINTF(...)                                                      \
  do {                                                                         \
    parcagpu::init_debug();                                                    \
    if (parcagpu::debug_enabled) {                                             \
      struct timespec ts;                                                      \
      clock_gettime(CLOCK_REALTIME, &ts);                                      \
      fprintf(stderr, "[%ld.%09ld] ", ts.tv_sec, ts.tv_nsec);                  \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

#define CALL(func_call, want, expr)                                            \
  do {                                                                         \
    auto ret = func_call;                                                      \
    if (ret != want) {                                                         \
      DEBUG_PRINTF("%s call failed, error code: %d\n", #func_call, ret);       \
      return expr;                                                             \
    }                                                                          \
  } while (0)

#define CALL_MSPTI_VOID(mspti_func, ...)                                       \
  CALL(mspti_func(__VA_ARGS__), MSPTI_SUCCESS, )
#define CALL_MSPTI_FALSE(mspti_func, ...)                                      \
  CALL(mspti_func(__VA_ARGS__), MSPTI_SUCCESS, false)

namespace parcagpu {
// Global correlation tracking instances
static CorrelationFilter g_correlationFilter;

// Debug logging control
bool debug_enabled = false;
void init_debug() {
  static bool initialized = false;
  if (initialized)
    return;
  debug_enabled = getenv("COLAGPU_DEBUG") != nullptr;
  initialized = true;
}
} // namespace parcagpu

namespace parcagpu {

// Simplified profiler using Proton's patterns
class MsptiProfiler : public proton::Singleton<MsptiProfiler> {
public:
  MsptiProfiler() { DEBUG_PRINTF("[COLAGPU] Initializing ParcaGPUProfiler\n"); }

  ~MsptiProfiler() { cleanup(); }

  bool initialize() {
    if (initialized.exchange(true))
      return true; // Already initialized
    CALL_MSPTI_FALSE(msptiSubscribe, &subscriber, callbackHandler, nullptr);
    // 主计算单元，执行神经网络算子（矩阵乘、卷积等）。最常见、最核心的 launch
    CALL_MSPTI_FALSE(msptiEnableCallback, 1, subscriber,
                     MSPTI_CB_DOMAIN_RUNTIME, MSPTI_CBID_RUNTIME_LAUNCH);
    // Ascend 上的专用 CPU 核，处理不适合 AI Core
    // 的算子（如非矩阵运算、控制流逻辑等）
    CALL_MSPTI_FALSE(msptiEnableCallback, 1, subscriber,
                     MSPTI_CB_DOMAIN_RUNTIME, MSPTI_CBID_RUNTIME_AICPU_LAUNCH);
    // AI Vector核心
    CALL_MSPTI_FALSE(msptiEnableCallback, 1, subscriber,
                     MSPTI_CB_DOMAIN_RUNTIME, MSPTI_CBID_RUNTIME_AIV_LAUNCH);
    // AI FFT加速器
    CALL_MSPTI_FALSE(msptiEnableCallback, 1, subscriber,
                     MSPTI_CB_DOMAIN_RUNTIME, MSPTI_CBID_RUNTIME_FFTS_LAUNCH);
    // 回退到 Host CPU 执行的算子。某些不适配 NPU 的算子会 fallback 到这里
    // CALL_MSPTI_FALSE(msptiEnableCallback, 1, subscriber,
    // MSPTI_CB_DOMAIN_RUNTIME, MSPTI_CBID_RUNTIME_CPU_LAUNCH);
    return true;
  }

  void cleanup() {
    if (!initialized.exchange(false)) {
      return; // Already cleaned up
    }
    DEBUG_PRINTF("[COLAGPU] Cleanup started\n");
  }

private:
  std::atomic<bool> initialized{false};

  static constexpr size_t AlignSize = 8;
  static constexpr size_t BufferSize = 128 * 1024;
  static void allocBuffer(uint8_t **buffer, size_t *bufferSize,
                          size_t *maxNumRecords) {
    if (!COLAGPU_API_CORRELATION_ENABLED()) {
      *buffer = nullptr;
      return;
    }
    *buffer = static_cast<uint8_t *>(aligned_alloc(AlignSize, BufferSize));
    if (*buffer == nullptr) {
      DEBUG_PRINTF("[COLAGPU] ERROR: aligned_alloc failed\n");
      return;
    }
    *bufferSize = BufferSize;
    *maxNumRecords = 0;
    DEBUG_PRINTF("[PARCAGPU:allocBuffer] Allocated buffer at %p size %zu\n",
                 *buffer, *bufferSize);
  }

  static void completeBuffer(uint8_t *buffer, size_t size, size_t validSize) {
    if (validSize <= 0) {
      free(buffer);
      DEBUG_PRINTF("validSize is invalid");
      return;
    }
    DEBUG_PRINTF("[COLAGPU] completeBuffer called: buffer=%p validSize=%zu\n",
                 buffer, validSize);

    msptiActivity *pRecord = nullptr;
    msptiResult status =
        msptiActivityGetNextRecord(buffer, validSize, &pRecord);
    for (; status == MSPTI_SUCCESS;
         status = msptiActivityGetNextRecord(buffer, validSize, &pRecord)) {
      msptiActivityKind kind = pRecord->kind;
      switch (kind) {
      case MSPTI_ACTIVITY_KIND_KERNEL: {
        msptiActivityKernel *k =
            reinterpret_cast<msptiActivityKernel *>(pRecord);
        // Regular kernel - check and remove from correlation filter
        uint32_t kernelTid = 0;
        bool shouldEmit =
            g_correlationFilter.check_and_remove(k->correlationId, &kernelTid);

        if (!shouldEmit) {
          DEBUG_PRINTF("[COLAGPU] Filtered kernel activity: correlationId=%u "
                       "(not in filter)\n",
                       k->correlationId);
          // Skip both KERNEL_EXECUTED and activity_batch push. Without this,
          // the eBPF activity_batch consumer would emit a kernel_event for
          // every rate-limited launch, producing orphan entries in
          // parca-agent's timesAwaitingTraces map (no cuda_correlation USDT was
          // emitted for these correlation IDs, so no trace will ever arrive to
          // match them).
          break;
        }

        DEBUG_PRINTF(
            "[COLAGPU] Kernel activity: name=%s, correlationId=%u, "
            "deviceId=%u, streamId=%u, start=%lu, end=%lu, duration=%lu ns\n",
            k->name, k->correlationId, k->ds.deviceId, k->ds.streamId, k->start,
            k->end, k->end - k->start);

        // Emit USDT probe for kernel execution
        COLAGPU_KERNEL_EXECUTED(k->start, k->end, k->correlationId,
                                k->ds.deviceId, k->ds.streamId, 0, 0, k->name);
        // Note: kernelTid captured via check_and_remove() above; exposed via
        // the ACTIVITY_BATCH USDT probe on the eBPF side.
        (void)kernelTid;
        break;
      }
      default:
        break;
      }
    }
    if (status != MSPTI_ERROR_MAX_LIMIT_REACHED)
      DEBUG_PRINTF("Consume data fail, error is %d", status);
    free(buffer);
  }

  msptiSubscriberHandle subscriber;
  static std::atomic<bool> activityEnabled;
  static void callbackHandler(void *pUserData, msptiCallbackDomain domain,
                              msptiCallbackId callbackId,
                              const msptiCallbackData *pCallbackInfo) {
    (void)pUserData;
    // 通过LD_PRELOAD加载，需要等待发起真实的NPU调用才能确保设备正常初始化，这里的callback才会生效。不然有概率触发崩溃
    if (!activityEnabled.exchange(true)) {
      CALL_MSPTI_VOID(msptiActivityRegisterCallbacks, allocBuffer,
                      completeBuffer);
      CALL_MSPTI_VOID(msptiActivityEnable, MSPTI_ACTIVITY_KIND_KERNEL);
      DEBUG_PRINTF("[COLAGPU] Device-side activity tracing enabled\n");
    }

    if (!COLAGPU_API_CORRELATION_ENABLED()) {
      g_correlationFilter.trim(0);
      return;
    }
    if (domain != MSPTI_CB_DOMAIN_RUNTIME || pCallbackInfo == nullptr ||
        pCallbackInfo->callbackSite != MSPTI_API_EXIT)
      return;
    COLAGPU_API_CORRELATION(pCallbackInfo->correlationId, callbackId,
                            pCallbackInfo->functionName);
    g_correlationFilter.insert(pCallbackInfo->correlationId,
                               (uint32_t)syscall(SYS_gettid));
    // Prune stale entries if the filter grows too large
    if (g_correlationFilter.size() > 10000) {
      uint32_t threshold = pCallbackInfo->correlationId > 5000
                               ? pCallbackInfo->correlationId - 5000
                               : 0;
      g_correlationFilter.trim(threshold);
    }
  }
};

std::atomic<bool> MsptiProfiler::activityEnabled{false};

} // namespace parcagpu

__attribute__((constructor)) void LoadProfiler() {
  DEBUG_PRINTF("[COLAGPU] InitializeInjection called\n");
  try {
    auto &profiler = parcagpu::MsptiProfiler::instance();
    if (!profiler.initialize()) {
      return; // Return 0 on failure, but don't break injection
    }
    atexit([]() {
      try {
        parcagpu::MsptiProfiler::instance().cleanup();
      } catch (const std::exception &e) {
        fprintf(stderr, "[COLAGPU] cleanup caught: %s\n", e.what());
      } catch (...) {
        fprintf(stderr, "[COLAGPU] cleanup caught unknown exception\n");
      }
    });
    return;
  } catch (const std::exception &e) {
    fprintf(stderr, "[COLAGPU] InitializeInjection caught: %s\n", e.what());
    return;
  } catch (...) {
    fprintf(stderr, "[COLAGPU] InitializeInjection caught unknown exception\n");
    return;
  }
}