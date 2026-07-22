// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "activity.h"

// Include proton headers
#include "Driver/GPU/CuptiApi.h"
#include "Profiler/Cupti/CuptiCallbacks.h"
#include "Utility/Singleton.h"
#include "correlation_filter.h"
#include "env_config.h"
#include "pc_sampling.h"
#include "token_bucket.h"

constexpr std::array<CUpti_CallbackId, 4> kSynchronizeCallbacks = {
    CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize,
    CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize,
    CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize,
    CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize_ptsz};

// Runtime memcpy CBIDs — enable so the callback handler can capture pid/tid
// on every cudaMemcpy/cudaMemcpyAsync EXIT, then bridge them to activity
// records via MemcpyCorrelationMap. Without these callbacks firing, the map
// is never populated and pid/tid in memcpy events stay 0.
constexpr std::array<CUpti_CallbackId, 4> kMemcpyRuntimeCallbacks = {
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_ptds_v7000,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020,
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_ptsz_v7000};

static bool isMemcpyRuntimeCbid(CUpti_CallbackId cbid) {
  for (auto id : kMemcpyRuntimeCallbacks) {
    if (id == cbid)
      return true;
  }
  return false;
}

namespace parcagpu {

// Debug logging control
bool debug_enabled = false;

// Global correlation tracking instances
static CorrelationFilter g_correlationFilter;
static GraphCorrelationMap g_graphCorrelationMap;
static MemcpyCorrelationMap g_memcpyCorrelationMap;
static std::atomic<uint32_t> g_bufferCycle{0};

// Thread-local tracking: store correlation ID from runtime ENTER
// so we can skip driver EXIT probe when it matches (driver calls happen under
// runtime calls)
thread_local uint32_t runtimeEnterCorrelationId = 0;

// Thread-local rate limiter for callback probes (default 100/sec,
// configurable via COLAGPU_RATE_LIMIT).
thread_local TokenBucket callbackLimiter(100.0);

// ---------------------------------------------------------------------------
// PC sampling probabilistic control.
//
// Sampling is gated by a per-thread interval + dice-roll mechanism: at most
// once per kPCSamplingIntervalNs, roll against probability; if it hits, open
// a sampling window that stays active until the next interval boundary.
//
// The user-facing knob is COLAGPU_PC_SAMPLING_RATE (samples/sec); a
// process-wide controller adjusts the dice-roll probability over time so the
// observed sample rate converges on the target. Internally the controller
// reads `samplesTotal` (incremented from pc_sampling.cpp on every batch) and
// recalibrates probability every kPCControlPeriodNs based on the rate
// observed since the last update.
// ---------------------------------------------------------------------------

// 30 ms — short enough to give ~33 dice rolls/sec (tighter rate variance,
// faster response to workload phase changes), long enough that CUPTI
// start/stop cost (~25 us measured) is amortized to <0.1% of wall time.
static constexpr uint64_t kPCSamplingIntervalNs = 30'000'000ULL;
// How often the controller recalibrates probability based on observed rate.
static constexpr uint64_t kPCControlPeriodNs = 5'000'000'000ULL;
// Don't react to <25% rate error (avoids oscillating on noise).
static constexpr double kPCControlTolerance = 0.25;
// Symmetric step clamp at sqrt(2). Larger steps (2x or 4x) amplify
// single-window sampling noise into multi-update oscillations: one
// 30ms window happening to land in an idle phase shows few samples,
// the controller over-corrects, then the next busy phase shows many.
// sqrt(2) keeps each adjustment small enough that the eventual
// equilibrium sits within the tolerance band even under bursty signal.
static constexpr double kPCControlStepShrink = 1.41421356;
static constexpr double kPCControlStepGrow = 1.41421356;
static constexpr double kPCProbMin = 0.001;
static constexpr double kPCProbMax = 1.0;
// Initial probability *at the default target rate*. Higher values waste
// samples on kernel-dense workloads (FNS-class: 5K launches/sec); lower
// values starve kernel-sparse workloads (a few launches/sec) of dice
// rolls until the controller grows it. 0.02 is a tested compromise at
// targetRate=100. Real initial probability scales with targetRate (see
// init_debug) so callers asking for high rates start sampling
// immediately instead of waiting many control periods to climb.
static constexpr double kPCInitialProbabilityAtDefaultRate = 0.02;
// Default target rate when COLAGPU_PC_SAMPLING_RATE is unset.
static constexpr double kPCDefaultTargetRate = 100.0;

struct PCRateController {
  double targetRate = kPCDefaultTargetRate; // immutable after init
  std::atomic<double> probability{kPCInitialProbabilityAtDefaultRate};
  std::atomic<uint64_t> samplesTotal{0};
  std::atomic<uint64_t> lastCheckNs{0};
  std::atomic<uint64_t> lastCheckTotal{0};
};
static PCRateController g_pcController;

// Per-thread sampling state.
struct PCSamplingState {
  bool active = false;        // Currently sampling
  uint64_t windowStartNs = 0; // When the current window opened
  uint64_t lastCheckNs = 0;   // Last time we rolled the dice
  unsigned int rngSeed = 0;   // Thread-local RNG state
};
thread_local PCSamplingState g_pcSamplingState;

static uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// Seed the per-thread RNG lazily.
static void ensureRngSeeded(PCSamplingState &s) {
  if (s.rngSeed == 0) {
    uint64_t t = nowNs();
    s.rngSeed = (unsigned int)(t ^ (uintptr_t)&s);
    if (s.rngSeed == 0)
      s.rngSeed = 1;
  }
}

static double threadRandom(PCSamplingState &s) {
  ensureRngSeeded(s);
  return (double)rand_r(&s.rngSeed) / RAND_MAX;
}

// Called from pc_sampling.cpp:processPCSamplingData() once per CUPTI batch
// with the sum of distinct (PC, stallReason) pairs that had non-zero samples
// — the unit the agent emits as a gpu_pc record on the receive side, and the
// right rate to steer the controller with.
void recordPCSamples(uint64_t n) {
  g_pcController.samplesTotal.fetch_add(n, std::memory_order_relaxed);
}

// Adjust controller.probability to converge on targetRate. Cheap to call —
// returns immediately unless kPCControlPeriodNs has elapsed since the last
// adjustment. Safe under concurrent calls (CAS on lastCheckNs).
static void controllerMaybeUpdate() {
  if (g_pcController.targetRate <= 0.0)
    return;
  uint64_t now = nowNs();
  uint64_t last = g_pcController.lastCheckNs.load(std::memory_order_relaxed);
  if (now - last < kPCControlPeriodNs)
    return;
  // Only one thread per period actually performs the update.
  if (!g_pcController.lastCheckNs.compare_exchange_strong(
          last, now, std::memory_order_acq_rel))
    return;
  uint64_t total = g_pcController.samplesTotal.load(std::memory_order_relaxed);
  uint64_t lastTotal =
      g_pcController.lastCheckTotal.exchange(total, std::memory_order_acq_rel);
  uint64_t delta = total - lastTotal;
  uint64_t elapsed = now - last;
  if (elapsed == 0)
    return;
  double observedRate = (double)delta * 1e9 / (double)elapsed;
  double err =
      (observedRate - g_pcController.targetRate) / g_pcController.targetRate;
  if (std::abs(err) <= kPCControlTolerance)
    return;
  double ratio = g_pcController.targetRate / std::max(observedRate, 1e-3);
  ratio = std::clamp(ratio, 1.0 / kPCControlStepShrink, kPCControlStepGrow);
  double oldP = g_pcController.probability.load(std::memory_order_relaxed);
  double newP = std::clamp(oldP * ratio, kPCProbMin, kPCProbMax);
  g_pcController.probability.store(newP, std::memory_order_relaxed);
  DEBUG_PRINTF("[COLAGPU] PC rate controller: observed=%.2f target=%.2f "
               "old_p=%.5f new_p=%.5f\n",
               observedRate, g_pcController.targetRate, oldP, newP);
}

void init_debug() {
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    debug_enabled = getenv("COLAGPU_DEBUG") != nullptr;
    const char *rateEnv = getenv("COLAGPU_RATE_LIMIT");
    if (rateEnv != nullptr) {
      double rate = atof(rateEnv);
      if (rate > 0) {
        callbackLimiter.setRate(rate);
      }
    }

    const char *targetRateEnv = getenv("COLAGPU_PC_SAMPLING_RATE");
    if (targetRateEnv) {
      double r = atof(targetRateEnv);
      if (r > 0.0)
        g_pcController.targetRate = r;
    }
    // Scale initial probability with target rate so high-target requests
    // start sampling immediately. The controller fires only when a
    // sampling window closes; if the workload launches kernels rarely
    // (vortex-class: ~1 launch/sec) and the initial probability is too
    // low for a window to open, the controller starves and can never
    // climb. Scaling avoids that for the common "user wants lots of
    // samples on a sparse workload" case.
    {
      double scaled = kPCInitialProbabilityAtDefaultRate *
                      (g_pcController.targetRate / kPCDefaultTargetRate);
      double initP = std::clamp(scaled, kPCProbMin, kPCProbMax);
      g_pcController.probability.store(initP, std::memory_order_relaxed);
    }
    g_pcController.lastCheckNs.store(nowNs(), std::memory_order_relaxed);

    validateEnvVars();
    initialized = true;
  }
}

// Out-of-line USDT probe site for activity batches.
// Single call site ensures one probe location in the ELF .note.stapsdt section.
} // namespace parcagpu

namespace parcagpu {

// Simplified profiler using Proton's patterns
class CuptiProfiler : public proton::Singleton<CuptiProfiler> {
public:
  CuptiProfiler() { DEBUG_PRINTF("[COLAGPU] Initializing ParcaGPUProfiler\n"); }

  ~CuptiProfiler() { cleanup(); }

  bool initialize() {
    if (initialized.exchange(true)) {
      return true; // Already initialized
    }

    DEBUG_PRINTF("[COLAGPU] Starting initialization\n");

    // Check if PC sampling is supported
    pcSamplingEnabled = parcagpu::PCSampling::isSupported();
    if (pcSamplingEnabled) {
      DEBUG_PRINTF("[COLAGPU] PC sampling enabled (serialized mode)\n");
    } else {
      DEBUG_PRINTF(
          "[COLAGPU] PC sampling disabled, using kernel activity only\n");
    }

    // Subscribe to callbacks
    auto result =
        proton::cupti::subscribe<true>(&subscriber, callbackHandler, nullptr);
    if (result != CUPTI_SUCCESS) {
      DEBUG_PRINTF("[COLAGPU] Failed to subscribe to callbacks: error %d\n",
                   result);
      return false;
    }

    // Enable runtime and driver API callbacks (using Proton's utilities)
    proton::setRuntimeCallbacks(subscriber, /*enable=*/true);
    proton::setLaunchCallbacks(subscriber, /*enable=*/true);

    // Enable resource callbacks only if PC sampling is enabled
    if (pcSamplingEnabled) {
      proton::setResourceCallbacks(subscriber, /*enable=*/true);
    }

    // Enable synchronize driver API callbacks for sync tracking
    for (auto cbId : kSynchronizeCallbacks) {
      result = proton::cupti::enableCallback<true>(
          /*enable=*/1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, cbId);
      if (result != CUPTI_SUCCESS) {
        DEBUG_PRINTF("[COLAGPU] Failed to enableCallback %d: error %d\n", cbId,
                     result);
        return false;
      } else {
        DEBUG_PRINTF("[COLAGPU] enable driver callback %d\n", cbId);
      }
    }

    // Enable runtime memcpy callbacks so pid/tid can be captured for
    // memcpy activity record correlation.
    for (auto cbId : kMemcpyRuntimeCallbacks) {
      result = proton::cupti::enableCallback<true>(
          /*enable=*/1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, cbId);
      if (result != CUPTI_SUCCESS) {
        DEBUG_PRINTF(
            "[COLAGPU] Failed to enable memcpy callback %d: error %d\n", cbId,
            result);
        return false;
      } else {
        DEBUG_PRINTF("[COLAGPU] enable runtime memcpy callback %d\n", cbId);
      }
    }

    // Register activity buffer callbacks (using Proton's pattern)
    result = proton::cupti::activityRegisterCallbacks<true>(allocBuffer,
                                                            completeBuffer);
    if (result != CUPTI_SUCCESS) {
      DEBUG_PRINTF(
          "[COLAGPU] Failed to register activity callbacks: error %d\n",
          result);
      return false;
    }

    // Enable activity kinds via loop — uses <false> (non-throwing) so one
    // failure doesn't abort the rest.
    std::map<CUpti_ActivityKind, std::string> activities = {
        {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, "CONCURRENT_KERNEL"},
        {CUPTI_ACTIVITY_KIND_DRIVER, "DRIVER"},
        {CUPTI_ACTIVITY_KIND_RUNTIME, "RUNTIME"},
        {CUPTI_ACTIVITY_KIND_MEMCPY, "MEMCPY"},
        {CUPTI_ACTIVITY_KIND_MEMCPY2, "MEMCPY2(P2P)"},
    };
    for (const auto &[kind, name] : activities) {
      if (auto r = proton::cupti::activityEnable<false>(kind);
          r != CUPTI_SUCCESS) {
        DEBUG_PRINTF("[COLAGPU] Failed to enable %s activity: error %d\n",
                     name.c_str(), r);
      } else {
        DEBUG_PRINTF("[COLAGPU] Enabled %s activity\n", name.c_str());
      }
    }

    DEBUG_PRINTF("[COLAGPU] Successfully initialized CUPTI callbacks\n");
    return true;
  }

  void cleanup() {
    if (!initialized.exchange(false)) {
      return; // Already cleaned up
    }

    DEBUG_PRINTF("[COLAGPU] Cleanup started\n");

    // PC sampling data is drained in finalize() during CONTEXT_DESTROY_STARTING
    // when the CUDA context is still valid. By the time cleanup() runs, the
    // context may already be dead, so we don't drain here.

    // Disable all callbacks
    if (subscriber) {
      proton::setRuntimeCallbacks(subscriber, /*enable=*/false);
      proton::setLaunchCallbacks(subscriber, /*enable=*/false);

      for (auto cbId : kSynchronizeCallbacks) {
        proton::cupti::enableCallback<false>(/*enable=*/0, subscriber,
                                             CUPTI_CB_DOMAIN_DRIVER_API, cbId);
      }
      for (auto cbId : kMemcpyRuntimeCallbacks) {
        proton::cupti::enableCallback<false>(/*enable=*/0, subscriber,
                                             CUPTI_CB_DOMAIN_RUNTIME_API, cbId);
      }
      if (pcSamplingEnabled) {
        proton::setResourceCallbacks(subscriber, /*enable=*/false);
      }
    }

    // Cleanup runs from atexit and may also reach us via a libcupti callback
    // whose caller wasn't built with C++ EH; throwing from here aborts the
    // process. Use <false> + log on the cleanup-path calls.
    if (auto r = proton::cupti::activityFlushAll<false>(
            CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
        r != CUPTI_SUCCESS) {
      DEBUG_PRINTF("[COLAGPU] activityFlushAll failed: %d\n", r);
    }

    // Disable all activity kinds (mirrors the enable loop above)
    for (CUpti_ActivityKind kind :
         {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_DRIVER,
          CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_MEMCPY,
          CUPTI_ACTIVITY_KIND_MEMCPY2}) {
      if (auto r = proton::cupti::activityDisable<false>(kind);
          r != CUPTI_SUCCESS) {
        DEBUG_PRINTF("[COLAGPU] activityDisable(%d) failed: %d\n", kind, r);
      }
    }

    if (subscriber) {
      if (auto r = proton::cupti::unsubscribe<false>(subscriber);
          r != CUPTI_SUCCESS) {
        DEBUG_PRINTF("[COLAGPU] unsubscribe failed: %d\n", r);
      }
      subscriber = nullptr;
    }

    DEBUG_PRINTF("[COLAGPU] Cleanup completed\n");
  }

private:
  std::atomic<bool> initialized{false};
  bool pcSamplingEnabled = false;
  CUpti_SubscriberHandle subscriber = nullptr;

  // PC sampling state — owned by this profiler, destroyed with it.
  parcagpu::PCSampling pcSampling;

  // Outstanding event counter for flushing
  size_t outstandingEvents = 0;

  // Buffer management - using Proton's pattern (static methods)
  // A kernel activity is around 224 bytes so a 128kb buffer
  // will hold ~500 activities, we want to flush regularly since
  // we are a continuous profiler so we don't need a huge buffer
  // like most CUPTI profilers.  Also a small size avoids malloc
  // just going to mmap every time so the allocator should cache
  // and re-use these for us.
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

  static void completeBuffer(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                             size_t size, size_t validSize) {
    CUpti_Activity *record = nullptr;
    int recordCount = 0;
    int filteredCount = 0;

    // Batch probe: collect ActivityEvent structs and pass them to
    // BPF/USDT every ACTIVITY_BATCH_SIZE records. Stack-allocated — no heap
    // allocation. The parcagpuActivityBatch function handles both individual
    // KERNEL_EXECUTED probes and the batch ACTIVITY_BATCH probe.
    ActivityEvent batchEvents[ACTIVITY_BATCH_SIZE];
    uint32_t batchCount = 0;

    DEBUG_PRINTF(
        "[COLAGPU] completeBuffer called: ctx=%p buffer=%p validSize=%zu\n",
        ctx, buffer, validSize);

    // Start a new buffer cycle for graph correlation tracking
    uint32_t cycle = g_bufferCycle.fetch_add(1);
    g_graphCorrelationMap.cycle_start(cycle);

    while (true) {
      CUptiResult result = proton::cupti::activityGetNextRecord<false>(
          buffer, validSize, &record);
      if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else if (result != CUPTI_SUCCESS) {
        DEBUG_PRINTF("[COLAGPU] Error reading activity record: error %d\n",
                     result);
        break;
      }

      recordCount++;

      ActivityEvent &evt = batchEvents[batchCount++];
      memset(&evt, 0, sizeof(evt));
      switch (record->kind) {
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      case CUPTI_ACTIVITY_KIND_KERNEL: {
        auto *k = reinterpret_cast<CUpti_ActivityKernel5 *>(record);

        // Check correlation filter - only emit probe if this kernel was sampled
        bool shouldEmit = false;
        if (k->graphId != 0) {
          // Graph kernel - check graph correlation map
          shouldEmit = g_graphCorrelationMap.check_and_mark_seen(
              k->correlationId, cycle);
        } else {
          // Regular kernel - check and remove from correlation filter
          shouldEmit = g_correlationFilter.check_and_remove(k->correlationId);
        }

        if (!shouldEmit) {
          filteredCount++;
          DEBUG_PRINTF("[COLAGPU] Filtered kernel activity: correlationId=%u "
                       "graphId=%u (not in filter)\n",
                       k->correlationId, k->graphId);
          // Skip both KERNEL_EXECUTED and activity_batch push. Without this,
          // the eBPF activity_batch consumer would emit a kernel_event for
          // every rate-limited launch, producing orphan entries in
          // parca-agent's timesAwaitingTraces map (no cuda_correlation USDT was
          // emitted for these correlation IDs, so no trace will ever arrive to
          // match them).
          break;
        }

        DEBUG_PRINTF("[COLAGPU] Kernel activity: graphId=%u graphNodeId=%lu "
                     "name=%s, correlationId=%u, deviceId=%u, "
                     "streamId=%u, start=%lu, end=%lu, duration=%lu ns\n",
                     k->graphId, k->graphNodeId, k->name, k->correlationId,
                     k->deviceId, k->streamId, k->start, k->end,
                     k->end - k->start);

        // Populate ActivityEvent for batch processing.
        // Only for kernel records that passed the correlation filter.
        // parcagpuActivityBatch handles both individual KERNEL_EXECUTED probes
        // and the batch ACTIVITY_BATCH probe in one call.
        evt.kind = ACTIVITY_KIND_KERNEL;
        evt.start = k->start;
        evt.end = k->end;
        evt.correlationId = k->correlationId;
        evt.deviceId = k->deviceId;
        evt.streamId = k->streamId;
        evt.graphId = k->graphId;
        evt.graphNodeId = k->graphNodeId;
        evt.name = k->name;
        break;
      }
      case CUPTI_ACTIVITY_KIND_RUNTIME:
      case CUPTI_ACTIVITY_KIND_DRIVER: {
        CUpti_ActivityAPI *api = reinterpret_cast<CUpti_ActivityAPI *>(record);
        bool is_launch;
        if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
          switch (api->cbid) {
          case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
          case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060:
          case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060:
          case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000:
            is_launch = true;
            break;
          default:
            is_launch = false;
            break;
          }
        } else {
          is_launch = proton::isLaunch(api->cbid);
        }
        if (!is_launch) {
          batchCount--;
          break;
        }
        evt.kind = ACTIVITY_KIND_HOST_API;
        evt.start = api->start;
        evt.end = api->end;
        // evt.streamId = api->cbid;
        evt.correlationId = api->correlationId;
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMCPY: {
        // Standard memory copy (H2D, D2H, D2D, H2A, A2H, etc.)
        auto *m = reinterpret_cast<CUpti_ActivityMemcpy6 *>(record);

        // Look up pid/tid via correlation ID. For runtime-launched memcpy
        // (cudaMemcpy*), the runtimeCorrelationId matches what we stored in
        // the callback handler. For driver-only memcpy (cuMemcpy*), the
        // correlationId matches.
        uint32_t lookupId = m->runtimeCorrelationId != 0
                                ? m->runtimeCorrelationId
                                : m->correlationId;
        MemcpyCorrelationMap::Info info{};
        bool found = g_memcpyCorrelationMap.check_and_remove(lookupId, &info);

        DEBUG_PRINTF(
            "[COLAGPU] Memcpy activity: copyKind=%u bytes=%lu "
            "deviceId=%u streamId=%u start=%lu end=%lu duration=%lu ns "
            "flags=%u correlationId=%u runtimeCorrId=%u pid=%u tid=%u\n",
            m->copyKind, m->bytes, m->deviceId, m->streamId, m->start, m->end,
            m->end - m->start, m->flags, m->correlationId,
            m->runtimeCorrelationId, found ? info.pid : 0,
            found ? info.tid : 0);

        evt.kind = ACTIVITY_KIND_MEMCPY;
        evt.start = m->start;
        evt.end = m->end;
        evt.correlationId = m->correlationId;
        evt.deviceId = m->deviceId;
        evt.streamId = m->streamId;
        evt.bytes = m->bytes;
        evt.copyKind = m->copyKind;
        evt.sync = (m->flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC) ? 0 : 1;
        evt.pid = found ? info.pid : 0;
        evt.tid = found ? info.tid : 0;
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMCPY2: {
        // Peer-to-peer memory copy
        auto *m = reinterpret_cast<CUpti_ActivityMemcpyPtoP4 *>(record);

        MemcpyCorrelationMap::Info info{};
        bool found =
            g_memcpyCorrelationMap.check_and_remove(m->correlationId, &info);

        DEBUG_PRINTF("[COLAGPU] Memcpy P2P activity: copyKind=%u bytes=%lu "
                     "deviceId=%u srcDeviceId=%u dstDeviceId=%u streamId=%u "
                     "start=%lu end=%lu duration=%lu ns flags=%u "
                     "correlationId=%u pid=%u tid=%u\n",
                     m->copyKind, m->bytes, m->deviceId, m->srcDeviceId,
                     m->dstDeviceId, m->streamId, m->start, m->end,
                     m->end - m->start, m->flags, m->correlationId,
                     found ? info.pid : 0, found ? info.tid : 0);

        evt.kind = ACTIVITY_KIND_MEMCPY;
        evt.start = m->start;
        evt.end = m->end;
        evt.correlationId = m->correlationId;
        evt.deviceId = m->deviceId;
        evt.streamId = m->streamId;
        evt.bytes = m->bytes;
        evt.copyKind = m->copyKind;
        evt.sync = (m->flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC) ? 0 : 1;
        evt.pid = found ? info.pid : 0;
        evt.tid = found ? info.tid : 0;
        break;
      }
      default: {
        batchCount--;
        DEBUG_PRINTF("[COLAGPU] Activity record %d: kind=%d\n", recordCount,
                     record->kind);
        break;
      }
      }

      if (batchCount >= ACTIVITY_BATCH_SIZE) {
        parcagpuActivityBatch(batchEvents, batchCount);
        batchCount = 0;
      }
    }

    // Flush remaining batch
    if (batchCount > 0) {
      parcagpuActivityBatch(batchEvents, batchCount);
    }

    // End cycle - cleanup completed graph entries
    g_graphCorrelationMap.cycle_end();

    DEBUG_PRINTF("[COLAGPU] Processed %d activity records (%d filtered) from "
                 "buffer %p\n",
                 recordCount, filteredCount, buffer);

    // Reset to 0 rather than decrement - one API callback can produce N
    // activities so decrementing by recordCount can cause underflow
    CuptiProfiler::instance().outstandingEvents = 0;

    // Free the buffer (Proton's pattern)
    std::free(buffer);
  }

  static void callbackHandler(void *userdata, CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid, const void *cbdata_void) {
    // libcupti invokes us from contexts whose callers weren't built with
    // C++ EH support, so any uncaught throw aborts the process. Treat the
    // whole body as untrusted and swallow exceptions at the boundary.
    try {
      auto &profiler = CuptiProfiler::instance();
      switch (domain) {
      case CUPTI_CB_DOMAIN_RESOURCE: {
        // Handle resource callbacks for PC sampling (only if enabled)
        if (!profiler.pcSamplingEnabled) {
          return;
        }

        const CUpti_ResourceData *resData =
            static_cast<const CUpti_ResourceData *>(cbdata_void);

        switch (cbid) {
        case CUPTI_CBID_RESOURCE_MODULE_LOADED: {
          const CUpti_ModuleResourceData *modData =
              static_cast<const CUpti_ModuleResourceData *>(
                  resData->resourceDescriptor);
          if (modData && modData->pCubin && modData->cubinSize > 0) {
            DEBUG_PRINTF("[COLAGPU] Module loaded: cubin=%p size=%zu\n",
                         modData->pCubin, modData->cubinSize);
            profiler.pcSampling.loadModule(modData->pCubin, modData->cubinSize);
          }
          break;
        }
        case CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING: {
          const CUpti_ModuleResourceData *modData =
              static_cast<const CUpti_ModuleResourceData *>(
                  resData->resourceDescriptor);
          if (modData && modData->pCubin && modData->cubinSize > 0) {
            DEBUG_PRINTF("[COLAGPU] Module unloading: cubin=%p size=%zu\n",
                         modData->pCubin, modData->cubinSize);
            profiler.pcSampling.unloadModule(modData->pCubin,
                                             modData->cubinSize);
          }
          break;
        }
        case CUPTI_CBID_RESOURCE_CONTEXT_CREATED: {
          CUcontext ctx = resData->context;
          DEBUG_PRINTF("[COLAGPU] Context created: %p\n", ctx);
          profiler.pcSampling.initialize(ctx);
          break;
        }
        case CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING: {
          CUcontext ctx = resData->context;
          DEBUG_PRINTF("[COLAGPU] Context destroying: %p\n", ctx);
          profiler.pcSampling.finalize(ctx);
          break;
        }
        default:
          break;
        }
        break;
      }
      case CUPTI_CB_DOMAIN_DRIVER_API: {
        if (std::find(std::begin(kSynchronizeCallbacks),
                      std::end(kSynchronizeCallbacks),
                      cbid) != std::end(kSynchronizeCallbacks)) {
          static thread_local uint64_t syncEnterNs = 0;
          const CUpti_CallbackData *cbdata =
              static_cast<const CUpti_CallbackData *>(cbdata_void);
          if (cbdata->callbackSite == CUPTI_API_ENTER) {
            syncEnterNs = nowNs();
          } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
            uint64_t exitNs = nowNs();

            const char *name =
                cbdata->functionName ? cbdata->functionName : "(unknown)";
            DEBUG_PRINTF("[COLAGPU] Synchronize: cbid=%u, duration=%lu ns, "
                         "func=%s, correlationId=%u\n",
                         cbid, exitNs - syncEnterNs, name,
                         cbdata->correlationId);
            if (COLAGPU_API_SYNCHRONIZE_ENABLED()) {
              COLAGPU_API_SYNCHRONIZE(syncEnterNs, exitNs, name);
            }
          }
          break;
        }
        // Non-synchronize DRIVER_API falls through to RUNTIME_API handling
      }
      case CUPTI_CB_DOMAIN_RUNTIME_API: {
        // Handle both Runtime and Driver API callbacks
        const CUpti_CallbackData *cbdata =
            static_cast<const CUpti_CallbackData *>(cbdata_void);
        uint32_t correlationId = cbdata->correlationId;

        // PC sampling windows are aligned to kernel-launch boundaries:
        // start on a launch ENTER, stop on a launch EXIT. The decision to
        // open a window is still time + probability gated; the launch CBIDs
        // just determine when the boundaries fire.
        const bool isKernelLaunchCb =
            domain == CUPTI_CB_DOMAIN_DRIVER_API && proton::isLaunch(cbid);

        // ENTER: open a sampling window on launch boundary if interval+prob
        // hits.
        if (cbdata->callbackSite == CUPTI_API_ENTER) {
          if (domain == CUPTI_CB_DOMAIN_RUNTIME_API)
            runtimeEnterCorrelationId = correlationId;

          if (profiler.pcSamplingEnabled) {
            if (isKernelLaunchCb) {
              auto &st = g_pcSamplingState;
              uint64_t now = nowNs();

              if (!st.active &&
                  (now - st.lastCheckNs >= kPCSamplingIntervalNs)) {
                st.lastCheckNs = now;
                const double p =
                    g_pcController.probability.load(std::memory_order_relaxed);
                if (threadRandom(st) < p) {
                  st.active = true;
                  st.windowStartNs = now;
                  profiler.pcSampling.start(cbdata->context);
                }
              }
            }

            profiler.pcSampling.emitMetadata();
          }
          return;
        }

        // Process on EXIT to avoid adding latency to GPU launch
        if (cbdata->callbackSite != CUPTI_API_EXIT) {
          return;
        }
        if (domain == CUPTI_CB_DOMAIN_RUNTIME_API && COLAGPU_ERROR_ENABLED()) {
          cudaError_t *err =
              reinterpret_cast<cudaError_t *>(cbdata->functionReturnValue);
          if (*err != cudaSuccess) {
            static auto cudaGetErrorName = (const char *(*)(cudaError_t))dlsym(
                RTLD_DEFAULT, "cudaGetErrorName");
            const char *message =
                cudaGetErrorName ? cudaGetErrorName(*err) : "Unknown";
            const char *component = "api";
            fireError(*(int32_t *)(err), message, component);
          }
        }

        // EXIT: while a sampling window is open, drain CUPTI's host staging
        // buffer on every CUDA API EXIT — not just launch EXITs. Empty drains
        // are cheap (single API call returning 0 PCs) and missed drains lose
        // samples (CUPTI_ERROR_OUT_OF_MEMORY when staging fills). Window close
        // is still kernel-launch aligned: only check elapsed-time on a launch
        // EXIT, so the window starts and ends at kernel boundaries.
        if (profiler.pcSamplingEnabled) {
          auto &st = g_pcSamplingState;

          if (st.active) {
            if (isKernelLaunchCb &&
                (nowNs() - st.windowStartNs >= kPCSamplingIntervalNs)) {
              profiler.pcSampling.stop(cbdata->context);
              st.active = false;
              controllerMaybeUpdate();
            } else {
              profiler.pcSampling.collectData(cbdata->context);
            }
          }

          profiler.pcSampling.emitMetadata();
        }

        // Skip correlation/rate-limiter work when no profiler is attached.
        if (!COLAGPU_API_CORRELATION_ENABLED())
          return;

        const char *name =
            cbdata->symbolName ? cbdata->symbolName : cbdata->functionName;
        int signedCbid;

        if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
          // Skip if this driver call is under a runtime call (same correlation
          // ID)
          if (correlationId == runtimeEnterCorrelationId) {
            DEBUG_PRINTF("[COLAGPU] Skipping driver EXIT correlationId=%u - "
                         "runtime will handle\n",
                         correlationId);
            return;
          }
          // Pure driver call (no runtime wrapper) - use negative cbid
          signedCbid = -(int)cbid;
          DEBUG_PRINTF("[COLAGPU] Driver API callback: cbid=%d, "
                       "correlationId=%u, func=%s\n",
                       cbid, correlationId, name);
        } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
          signedCbid = (int)cbid;
          runtimeEnterCorrelationId = 0; // Clear after use
          DEBUG_PRINTF("[COLAGPU] Runtime API callback: cbid=%d, "
                       "correlationId=%u, func=%s\n",
                       cbid, correlationId, name);

          // Capture pid/tid for memcpy activity correlation: only runtime
          // cudaMemcpy/cudaMemcpyAsync callbacks need this; kernel launches
          // and other APIs are irrelevant.
          if (isMemcpyRuntimeCbid(cbid)) {
            g_memcpyCorrelationMap.insert(correlationId, (uint32_t)getpid(),
                                          (uint32_t)syscall(SYS_gettid));
            // Prune stale entries to bound memory.
            if (g_memcpyCorrelationMap.size() > 10000) {
              uint32_t threshold =
                  correlationId > 5000 ? correlationId - 5000 : 0;
              g_memcpyCorrelationMap.trim(threshold);
            }
            return;
          }
        } else {
          return;
        }

        // Check if this is a graph launch (never rate limit these)
        bool isGraphLaunch = false;
        if (signedCbid < 0) {
          // Driver API: cuGraphLaunch = 514, cuGraphLaunch_ptsz = 515
          int driverCbid = -signedCbid;
          isGraphLaunch =
              (driverCbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
               driverCbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz);
        } else {
          // Runtime API: cudaGraphLaunch = 311, cudaGraphLaunch_ptsz = 312
          isGraphLaunch =
              (signedCbid == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000 ||
               signedCbid ==
                   CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000);
        }

        // Rate limit probes using token bucket.  Skip rate limiting for graph
        // launches (they share one correlation ID across many kernels) and when
        // PC sampling is active (every kernel needs its correlation callback so
        // PC samples can be matched with CPU stacks on the agent side).
        if (!isGraphLaunch && !g_pcSamplingState.active) {
          if (!callbackLimiter.tryAcquire()) {
            DEBUG_PRINTF("[COLAGPU] Rate limited: skipping probe for "
                         "correlationId=%u\n",
                         correlationId);
            return;
          }
        }

        profiler.outstandingEvents++;
        // Emit USDT probe with signed cbid (negative for driver, positive for
        // runtime)
        COLAGPU_API_CORRELATION(correlationId, signedCbid, name);

        // Insert into correlation filter so we can match kernel activities
        // later
        if (isGraphLaunch) {
          g_graphCorrelationMap.insert(correlationId);
          DEBUG_PRINTF("[COLAGPU] Inserted correlationId=%u into graph map\n",
                       correlationId);
        } else {
          g_correlationFilter.insert(correlationId);
          DEBUG_PRINTF(
              "[COLAGPU] Inserted correlationId=%u into correlation filter\n",
              correlationId);
        }

        // Flush if too many events pile up
        if (profiler.outstandingEvents > 3000) {
          DEBUG_PRINTF("[COLAGPU] Flushing: outstandingEvents=%zu\n",
                       profiler.outstandingEvents);
          if (auto r = proton::cupti::activityFlushAll<false>(0);
              r != CUPTI_SUCCESS) {
            DEBUG_PRINTF("[COLAGPU] activityFlushAll failed: %d\n", r);
          }
          profiler.outstandingEvents = 0;
        }
        break;
      }
      default:
        break;
      }
    } catch (const std::exception &e) {
      fprintf(stderr, "[COLAGPU] callbackHandler caught: %s\n", e.what());
    } catch (...) {
      fprintf(stderr, "[COLAGPU] callbackHandler caught unknown exception\n");
    }
  }
};

} // namespace parcagpu

// CUPTI initialization function required for CUDA_INJECTION64_PATH.
// Called from the CUDA driver, which wasn't built with C++ EH; a throw out
// of here would abort the host process. Catch everything at the boundary.
extern "C" int InitializeInjection(void) {
  DEBUG_PRINTF("[COLAGPU] InitializeInjection called\n");
  try {
    auto &profiler = parcagpu::CuptiProfiler::instance();
    if (!profiler.initialize()) {
      return 0; // Return 0 on failure, but don't break injection
    }
    atexit([]() {
      try {
        parcagpu::CuptiProfiler::instance().cleanup();
      } catch (const std::exception &e) {
        fprintf(stderr, "[COLAGPU] cleanup caught: %s\n", e.what());
      } catch (...) {
        fprintf(stderr, "[COLAGPU] cleanup caught unknown exception\n");
      }
    });
    return 1;
  } catch (const std::exception &e) {
    fprintf(stderr, "[COLAGPU] InitializeInjection caught: %s\n", e.what());
    return 0;
  } catch (...) {
    fprintf(stderr, "[COLAGPU] InitializeInjection caught unknown exception\n");
    return 0;
  }
}
