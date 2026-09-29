#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

// USDT probes — must come before any header that might include <sys/sdt.h>,
// so that _SDT_HAS_SEMAPHORES is defined first.
#include "probes.h"

#include "activity.h"
#include "mspti.h"

#include "Utility/Singleton.h"
#include "correlation_filter.h"
#include "error.h"
#include "sampling.h"

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

namespace parcagpu {
// Global correlation tracking instances
static CorrelationFilter g_correlationFilter;
static MemcpyCorrelationMap g_memcpyCorrelationMap;

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

// Vendor-neutral synchronize kind, partitioned by backend: CUPTI uses
// 100-199, MSPTI 200-299 (see probes.d). MSPTI exposes a single synchronize
// runtime API (aclrtSynchronizeStream).
constexpr uint32_t kSyncKindAclrtSynchronizeStream = 200;

static uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// Runtime memcpy CBIDs — the callback handler captures pid/tid on these EXIT
// callbacks and bridges them to MEMCPY activity records via
// MemcpyCorrelationMap.
static bool isMemcpyRuntimeCbid(msptiCallbackId cbid) {
  return cbid == MSPTI_CBID_RUNTIME_MEMCPY ||
         cbid == MSPTI_CBID_RUNTIME_MEMCPY_ASYNC;
}

// Simplified profiler using Proton's patterns
class MsptiProfiler : public proton::Singleton<MsptiProfiler> {
public:
  MsptiProfiler() { DEBUG_PRINTF("[COLAGPU] Initializing ParcaGPUProfiler\n"); }

  ~MsptiProfiler() { cleanup(); }

  bool initialize() {
    if (initialized.exchange(true))
      return true; // Already initialized
    if (msptiSubscribe(&subscriber, callbackHandler, nullptr) != MSPTI_SUCCESS) 
      return false;

    // 注：CPU_LAUNCH 刻意不注册——回退到 Host CPU 执行的算子，某些不适配 NPU
    //     的算子会 fallback 到这里。
    static const msptiCallbackIdRuntime kRuntimeCbids[] = {
        MSPTI_CBID_RUNTIME_LAUNCH, // 主计算单元，执行神经网络算子（矩阵乘、卷积等）。最常见、最核心的
                                   // launch
        MSPTI_CBID_RUNTIME_AICPU_LAUNCH, //   专用 CPU 核，处理不适合 AI Core
                                         //   的算子（如非矩阵运算、控制流逻辑等）
        MSPTI_CBID_RUNTIME_AIV_LAUNCH,          //            AI Vector 核心
        MSPTI_CBID_RUNTIME_FFTS_LAUNCH,         //           AI FFT 加速器
        MSPTI_CBID_RUNTIME_STREAM_SYNCHRONIZED, //   用于 api_synchronize
                                                //   探针的同步耗时跟踪
        MSPTI_CBID_RUNTIME_MEMCPY,
        MSPTI_CBID_RUNTIME_MEMCPY_ASYNC, // 在 EXIT 回调里为 MEMCPY activity
                                         // 记录捕获 pid/tid
    };
    // 逐个注册需要跟踪的 Runtime API 回调：
    for (msptiCallbackIdRuntime cbid : kRuntimeCbids) {
      if (msptiEnableCallback(1, subscriber, MSPTI_CB_DOMAIN_RUNTIME, cbid) !=
          MSPTI_SUCCESS) {
        DEBUG_PRINTF("msptiEnableCallback call failed, cbid: %d\n", (int)cbid);
        return false;
      }
    }
    startThresholdPollThread();
    return true;
  }

  void cleanup() {
    if (!initialized.exchange(false)) {
      return; // Already cleaned up
    }
    DEBUG_PRINTF("[COLAGPU] Cleanup started\n");

    // Deliberately do NOT call msptiUnsubscribe / msptiActivityFlushAll /
    // msptiActivityDisable here. cleanup() runs from atexit(), by which time
    // the application (torch_npu/MindSpore) or the CANN runtime may already
    // have finalized the device/context; those teardown APIs touch dead device
    // state and segfault. This is the exit-time mirror of the lazy-init guard
    // above: MSPTI APIs are only safe while the device is live. The OS reclaims
    // the subscriber and activity buffers at process exit anyway — the only
    // cost is that in-flight activity records just before exit are dropped.
    stopThresholdPollThread();
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

    ActivityEvent kernelEvents[ACTIVITY_BATCH_SIZE];
    uint32_t kernelCount = 0;
    ActivityEvent hostEvents[ACTIVITY_BATCH_SIZE];
    uint32_t hostCount = 0;

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
        bool shouldEmit = g_correlationFilter.check_and_remove(
            (uint32_t)k->correlationId, &kernelTid);

        if (!shouldEmit) {
          DEBUG_PRINTF("[COLAGPU] Filtered kernel activity: correlationId=%u "
                       "(not in filter)\n",
                       (uint32_t)k->correlationId);
          // Skip kernel_timing push. Without this, the eBPF kernel_timing
          // consumer would emit a kernel_event for every rate-limited launch,
          // producing orphan entries in parca-agent's timesAwaitingTraces map
          // (no cuda_correlation USDT was emitted for these correlation IDs,
          // so no trace will ever arrive to match them).
          break;
        }

        DEBUG_PRINTF(
            "[COLAGPU] Kernel activity: name=%s, correlationId=%u, "
            "deviceId=%u, streamId=%u, start=%lu, end=%lu, duration=%lu ns\n",
            k->name, (uint32_t)k->correlationId, k->ds.deviceId, k->ds.streamId,
            k->start, k->end, k->end - k->start);

        // Emit kernel-timing event (batched, gated by kernel switch)
        ActivityEvent evt;
        memset(&evt, 0, sizeof(evt));
        evt.kind = ACTIVITY_KIND_KERNEL;
        evt.start = k->start;
        evt.end = k->end;
        evt.correlationId = (uint32_t)k->correlationId;
        evt.deviceId = k->ds.deviceId;
        evt.streamId = k->ds.streamId;
        evt.name = k->name;
        evt.tid = kernelTid;
        if (evt.start == 0 || evt.end == 0)
          break;
        kernelEvents[kernelCount++] = evt;
        if (kernelCount >= ACTIVITY_BATCH_SIZE) {
          parcagpuKernelTiming(kernelEvents, kernelCount);
          kernelCount = 0;
        }
        break;
      }
      case MSPTI_ACTIVITY_KIND_MEMCPY: {
        auto *m = reinterpret_cast<msptiActivityMemcpy *>(pRecord);

        MemcpyCorrelationMap::Info info{};
        bool found = g_memcpyCorrelationMap.check_and_remove(
            (uint32_t)m->correlationId, &info);

        DEBUG_PRINTF(
            "[COLAGPU] Memcpy activity: copyKind=%u bytes=%lu deviceId=%u "
            "streamId=%u start=%lu end=%lu duration=%lu ns isAsync=%u "
            "correlationId=%u tid=%u\n",
            m->copyKind, m->bytes, m->deviceId, m->streamId, m->start, m->end,
            m->end - m->start, (unsigned)m->isAsync, (uint32_t)m->correlationId,
            found ? info.tid : 0);

        // Map MSPTI copyKind to vendor-neutral values (see activity.h):
        // HTOD→100(H2D), DTOH→101(D2H), DTOD→101(D2D). HTOH/DEFAULT/UNKNOWN
        // are dropped (no vendor-neutral counterpart).
        uint16_t mappedCopyKind = 0;
        switch (m->copyKind) {
        case MSPTI_ACTIVITY_MEMCPY_KIND_HTOD:
          mappedCopyKind = 100;
          break;
        case MSPTI_ACTIVITY_MEMCPY_KIND_DTOH:
          mappedCopyKind = 101;
          break;
        case MSPTI_ACTIVITY_MEMCPY_KIND_DTOD:
          mappedCopyKind = 101;
          break;
        default:
          break;
        }
        if (mappedCopyKind == 0) {
          DEBUG_PRINTF(
              "[COLAGPU] Memcpy activity dropped: unsupported copyKind=%u\n",
              m->copyKind);
          break;
        }

        ActivityEvent evt;
        memset(&evt, 0, sizeof(evt));
        evt.kind = ACTIVITY_KIND_MEMCPY;
        evt.start = m->start;
        evt.end = m->end;
        evt.correlationId = (uint32_t)m->correlationId;
        evt.deviceId = m->deviceId;
        evt.streamId = m->streamId;
        evt.bytes = m->bytes;
        evt.copyKind = mappedCopyKind;
        evt.sync = m->isAsync ? 0 : 1;
        evt.tid = found ? info.tid : 0;
        if (evt.start == 0 || evt.end == 0)
          break;
        kernelEvents[kernelCount++] = evt;
        if (kernelCount >= ACTIVITY_BATCH_SIZE) {
          parcagpuKernelTiming(kernelEvents, kernelCount);
          kernelCount = 0;
        }
        break;
      }
      case MSPTI_ACTIVITY_KIND_RUNTIME_API: {
        auto *api = reinterpret_cast<msptiActivityApi *>(pRecord);

        // Host-side runtime API timing (ai-launch). The correlation filter
        // holds only sampled launches, so this non-removing check filters both
        // launch-vs-non-launch and sampled-vs-sampled-out in one test. The
        // kernel activity removes the same entry later via check_and_remove.
        if (!g_correlationFilter.check((uint32_t)api->correlationId))
          break;

        DEBUG_PRINTF("[COLAGPU] Host API activity: name=%s, correlationId=%u, "
                     "start=%lu, end=%lu, duration=%lu ns\n",
                     api->name ? api->name : "(unknown)",
                     (uint32_t)api->correlationId, api->start, api->end,
                     api->end - api->start);

        ActivityEvent evt;
        memset(&evt, 0, sizeof(evt));
        evt.kind = ACTIVITY_KIND_HOST_API;
        evt.start = api->start;
        evt.end = api->end;
        evt.correlationId = (uint32_t)api->correlationId;
        if (evt.start == 0 || evt.end == 0)
          break;
        hostEvents[hostCount++] = evt;
        if (hostCount >= ACTIVITY_BATCH_SIZE) {
          parcagpuHostTiming(hostEvents, hostCount);
          hostCount = 0;
        }
        break;
      }
      default:
        break;
      }
    }
    if (status != MSPTI_ERROR_MAX_LIMIT_REACHED)
      DEBUG_PRINTF("Consume data fail, error is %d", status);
    if (kernelCount > 0) {
      parcagpuKernelTiming(kernelEvents, kernelCount);
    }
    if (hostCount > 0) {
      parcagpuHostTiming(hostEvents, hostCount);
    }
    free(buffer);
  }

  msptiSubscriberHandle subscriber;
  static std::atomic<bool> activityEnabled;
  static void callbackHandler(void *pUserData, msptiCallbackDomain domain,
                              msptiCallbackId callbackId,
                              const msptiCallbackData *pCallbackInfo) {
    (void)pUserData;
    // 只订阅了 RUNTIME domain 的回调，其余 domain 直接丢弃
    if (domain != MSPTI_CB_DOMAIN_RUNTIME)
      return;
    // 通过LD_PRELOAD加载，需要等待发起真实的NPU调用才能确保设备正常初始化，这里的callback才会生效。不然有概率触发崩溃
    // exchange(true) 原子占位，保证只有一个线程进入初始化；失败则回退 false，
    // 下个回调重试（只有获胜线程能改回 false，无竞态）。
    if (!activityEnabled.exchange(true)) {
      // 注册设备侧回调，再启用消费的 activity 域（kernel/memcpy/runtime-api）。
      // 任一失败即回退 false，下个回调重试（只有获胜线程能改回 false，无竞态）。
      bool ok = msptiActivityRegisterCallbacks(allocBuffer, completeBuffer) ==
                MSPTI_SUCCESS;
      static const msptiActivityKind kActivityKinds[] = {
          MSPTI_ACTIVITY_KIND_KERNEL,
          MSPTI_ACTIVITY_KIND_MEMCPY,
          MSPTI_ACTIVITY_KIND_RUNTIME_API,
      };
      for (msptiActivityKind kind : kActivityKinds) {
        if (ok)
          ok = msptiActivityEnable(kind) == MSPTI_SUCCESS;
      }
      if (!ok) {
        activityEnabled.store(false);
      } else {
        DEBUG_PRINTF("[COLAGPU] Device-side activity tracing enabled\n");
      }
    }

    if (pCallbackInfo == nullptr)
      return;

    // Synchronize tracking: aclrtSynchronizeStream. Independent of the
    // correlation switch — gated only by the api_synchronize semaphore.
    // ENTER records the start timestamp, EXIT fires the probe (mirrors the
    // CUPTI backend's synchronize handling).
    if (callbackId == MSPTI_CBID_RUNTIME_STREAM_SYNCHRONIZED) {
      static thread_local uint64_t syncEnterNs = 0;
      if (pCallbackInfo->callbackSite == MSPTI_API_ENTER) {
        syncEnterNs = nowNs();
      } else if (pCallbackInfo->callbackSite == MSPTI_API_EXIT) {
        uint64_t exitNs = nowNs();
        DEBUG_PRINTF("[COLAGPU] Synchronize: duration=%lu ns, "
                     "correlationId=%u\n",
                     exitNs - syncEnterNs,
                     (uint32_t)pCallbackInfo->correlationId);
        if (COLAGPU_API_SYNCHRONIZE_ENABLED())
          COLAGPU_API_SYNCHRONIZE(syncEnterNs, exitNs,
                                  kSyncKindAclrtSynchronizeStream);
      }
      return;
    }

    // Error reporting: on RUNTIME EXIT, read the aclError return value and
    // fire the error probe for non-zero codes. Independent of the correlation
    // switch — gated only by the error semaphore (mirrors CUPTI).
    if (pCallbackInfo->callbackSite == MSPTI_API_EXIT &&
        COLAGPU_ERROR_ENABLED()) {
      const int *err =
          reinterpret_cast<const int *>(pCallbackInfo->functionReturnValue);
      if (err != nullptr && *err != 0) {
        int code = *err;
        const char *message = aclErrorName(code);
        char msgBuf[256];
        if (message)
          snprintf(msgBuf, sizeof(msgBuf), "%s", message);
        else
          snprintf(msgBuf, sizeof(msgBuf), "aclError_%d", code);

        const char *apiName = pCallbackInfo->functionName
                                  ? pCallbackInfo->functionName
                                  : "(unknown)";
        char compBuf[128];
        snprintf(compBuf, sizeof(compBuf), "api-%s", apiName);
        fireError(code, msgBuf, compBuf);
      }
    }

    if (!COLAGPU_API_CORRELATION_ENABLED()) {
      g_correlationFilter.trim(0);
      return;
    }
    if (pCallbackInfo->callbackSite != MSPTI_API_EXIT)
      return;

    // correlationId is uint64_t in MSPTI but uint32_t all the way downstream
    // (USDT probe, CorrelationFilter, Go consumer) — truncate explicitly.
    uint32_t correlationId = (uint32_t)pCallbackInfo->correlationId;

    // Memcpy pid/tid capture: bridge to MEMCPY activity via the correlation ID.
    // Gated on the kernel-timing semaphore — memcpy pid/tid only feeds
    // kernel_timing events. Sampled-out memcpys skip the insert, so their
    // activity records never match and never emit.
    
    if (!sampleRoll())
      return;
    if (isMemcpyRuntimeCbid(callbackId)) {
      if (!COLAGPU_KERNEL_TIMING_ENABLED())
        return;
      g_memcpyCorrelationMap.insert(correlationId, (uint32_t)getpid(),
                                    (uint32_t)syscall(SYS_gettid));
      // Prune stale entries to bound memory.
      if (g_memcpyCorrelationMap.size() > 10000) {
        uint32_t threshold = correlationId > 5000 ? correlationId - 5000 : 0;
        g_memcpyCorrelationMap.trim(threshold);
      }
      return;
    }

    COLAGPU_API_CORRELATION(correlationId, (int)callbackId,
                            pCallbackInfo->functionName);
    g_correlationFilter.insert(correlationId, (uint32_t)syscall(SYS_gettid));
    // Prune stale entries if the filter grows too large
    if (g_correlationFilter.size() > 10000) {
      uint32_t threshold = correlationId > 5000 ? correlationId - 5000 : 0;
      g_correlationFilter.trim(threshold);
    }
  }
};

std::atomic<bool> MsptiProfiler::activityEnabled{false};

} // namespace parcagpu

// ---------------------------------------------------------------------------
// aclInit interposition: MSPTI must be subscribed only AFTER the CANN runtime
// is initialized.
//
// libmspti.so (LD_PRELOAD'd alongside libcolamspti.so) has its own constructor
// that dlopen()s libprofapi.so. Because libcolamspti.so does NOT link against
// libmspti.so (it resolves the mspti* symbols from the preloaded library at run
// time), its constructor runs BEFORE libmspti.so's constructor — LD_PRELOAD
// objects are initialized in reverse order. Calling msptiSubscribe that early
// makes libmspti.so throw
//     "Failed to get function: ... from libprofapi.so"
// because libprofapi.so is not yet in the process.
//
// aclInit is the one CANN runtime entry point that libmspti.so does not
// interpose, so it is a reliable "runtime is ready" signal. Subscribe lazily,
// exactly once, right after the real aclInit succeeds.
extern "C" int aclInit(const char *configPath) {
  using AclInitFn = int (*)(const char *);
  static AclInitFn realAclInit =
      reinterpret_cast<AclInitFn>(dlsym(RTLD_NEXT, "aclInit"));
  if (realAclInit == nullptr) {
    // libascendcl.so not resolvable yet; nothing safe to chain to.
    return 1; // ACL_ERROR_INVALID_PARAM
  }
  int rc = realAclInit(configPath);
  if (rc == 0) { // ACL_SUCCESS
    try {
      parcagpu::MsptiProfiler::instance().initialize();
    } catch (const std::exception &e) {
      fprintf(stderr, "[COLAGPU] lazy initialize caught: %s\n", e.what());
    } catch (...) {
      fprintf(stderr, "[COLAGPU] lazy initialize caught unknown exception\n");
    }
  }
  return rc;
}

__attribute__((constructor)) void LoadProfiler() {
  DEBUG_PRINTF("[COLAGPU] LoadProfiler called\n");
  // Subscription is deferred to the aclInit interposition above (see the note
  // there on why calling msptiSubscribe from a load-time constructor fails).
  // Here we only install exit-time cleanup; cleanup() is a no-op if the
  // profiler was never initialized.
  atexit([]() {
    try {
      parcagpu::MsptiProfiler::instance().cleanup();
    } catch (const std::exception &e) {
      fprintf(stderr, "[COLAGPU] cleanup caught: %s\n", e.what());
    } catch (...) {
      fprintf(stderr, "[COLAGPU] cleanup caught unknown exception\n");
    }
  });
}