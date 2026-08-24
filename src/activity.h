#ifndef COLAGPU_ACTIVITY_H_
#define COLAGPU_ACTIVITY_H_

// USDT probes — must come before any header that might include <sys/sdt.h>,
// so that _SDT_HAS_SEMAPHORES is defined first.
#include "probes.h"

#include "cupti_activity.h"

static constexpr int ACTIVITY_BATCH_SIZE = 128;

typedef enum {
  ACTIVITY_KIND_KERNEL = 1,
  ACTIVITY_KIND_HOST_API = 2,
  ACTIVITY_KIND_MEMCPY = 3,
} ActivityKind;

typedef struct PACKED_ALIGNMENT {
  uint64_t kind;
  uint64_t start;
  uint64_t end;
  uint32_t correlationId;
  uint32_t deviceId;
  uint32_t streamId;
  uint32_t tid;

  uint64_t bytes;
  uint16_t
      copyKind;  // Vendor-neutral copy kind: H2D=100, D2H=101, D2D=101, P2P=103
  uint16_t sync; // 1=synchronous, 0=asynchronous

  //    kernel
  uint32_t graphId;
  uint64_t graphNodeId;
  const char *name;
} ActivityEvent;

__attribute__((noinline)) void parcagpuHostTiming(const ActivityEvent *events,
                                                  uint32_t count) {
  // Host-side timing (HOST-API / ai-launch). Gated by its own semaphore so the
  // agent can disable this probe independently, without touching kernel timing.
  if (!COLAGPU_HOST_TIMING_ENABLED())
    return;
  const void *batchPtrs[ACTIVITY_BATCH_SIZE];
  for (uint32_t i = 0; i < count; i++) {
    batchPtrs[i] = &events[i];
  }
  COLAGPU_HOST_TIMING(batchPtrs, count);
}

__attribute__((noinline)) void parcagpuKernelTiming(const ActivityEvent *events,
                                                    uint32_t count) {
  // GPU-side timing (KERNEL + MEMCPY / ai-execution + timeline). Gated by its
  // own semaphore so the agent can disable this probe independently.
  if (!COLAGPU_KERNEL_TIMING_ENABLED())
    return;
  const void *batchPtrs[ACTIVITY_BATCH_SIZE];
  for (uint32_t i = 0; i < count; i++) {
    batchPtrs[i] = &events[i];
  }
  COLAGPU_KERNEL_TIMING(batchPtrs, count);
}

#endif // COLAGPU_ACTIVITY_H_