// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

// Shared probabilistic sampling control for the CUPTI (NVIDIA) and MSPTI
// (Ascend) backends.
//
// The Go agent writes a per-mille threshold (0..1000) to
// /tmp/parcagpu.threshold inside each target process (through
// /proc/<pid>/root/tmp). A detached background thread polls that file and
// publishes the parsed value into g_sampleThreshold; callbacks read it via a
// relaxed load and roll the dice with a lock-free thread-local RNG, so the
// callback hot path never blocks.

#ifndef COLAGPU_SAMPLING_H_
#define COLAGPU_SAMPLING_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

namespace parcagpu {

// Probabilistic sampling threshold, per-mille in [0, 1000]: 1000 samples
// every event, 0 samples none. Published by the background reader thread
// below; read on the callback hot path via a relaxed load.
inline std::atomic<uint32_t> g_sampleThreshold{1000};

// Fixed path read by the background thread. The Go agent writes the same
// path through /proc/<pid>/root/tmp/parcagpu.threshold (it runs with
// hostPID), which resolves to this process's /tmp.
inline constexpr const char *kThresholdPath = "/tmp/parcagpu.threshold";

// Poll interval for the threshold reader thread.
inline constexpr auto kThresholdPollInterval = std::chrono::seconds(5);

// Thread-local xorshift32 RNG for the sampling dice-roll. std::rand() keeps
// global state and std::random_device may block — both are off-limits inside
// a callback. xorshift32 touches only thread-local state and is lock-free.
inline uint32_t fastRand() {
  static thread_local uint32_t state = 0;
  uint32_t x = state;
  if (x == 0) {
    x = static_cast<uint32_t>(syscall(SYS_gettid)) * 2654435761u ^ 0x9e3779b9u;
    if (x == 0)
      x = 0x9e3779b9u;
  }
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  state = x;
  return x;
}

// Roll the sampling dice; returns true when this event should be recorded.
inline bool sampleRoll() {
  uint32_t t = g_sampleThreshold.load(std::memory_order_relaxed);
  if (t >= 1000)
    return true;
  if (t == 0)
    return false;
  return (fastRand() % 1000u) < t;
}

// Detached reader thread and its shutdown flag.
inline std::atomic<bool> g_thresholdThreadRunning{false};
inline std::thread g_thresholdThread;

// Background reader: polls kThresholdPath every kThresholdPollInterval and
// publishes the parsed value into g_sampleThreshold. Blocking I/O lives here
// (never in a callback); a missing or malformed file keeps the previous value.
inline void thresholdPollThread() {
  while (g_thresholdThreadRunning.load(std::memory_order_relaxed)) {
    std::ifstream in(kThresholdPath);
    if (in.good()) {
      uint32_t v = 0;
      if (in >> v) {
        if (v > 1000)
          v = 1000;
        g_sampleThreshold.store(v, std::memory_order_relaxed);
      }
    }
    // Sleep in small slices so shutdown (which flips the running flag) takes
    // effect within ~100ms rather than waiting out a full interval.
    for (int i = 0;
         i < 50 && g_thresholdThreadRunning.load(std::memory_order_relaxed);
         ++i) {
      std::this_thread::sleep_for(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              kThresholdPollInterval) /
          50);
    }
  }
}

// Starts the detached threshold reader thread exactly once.
inline void startThresholdPollThread() {
  bool expected = false;
  if (!g_thresholdThreadRunning.compare_exchange_strong(expected, true))
    return; // already running
  g_thresholdThread = std::thread(thresholdPollThread);
  g_thresholdThread.detach();
}

// Signals the reader thread to stop. The thread is detached; it exits within
// ~100ms.
inline void stopThresholdPollThread() {
  g_thresholdThreadRunning.store(false, std::memory_order_relaxed);
}

} // namespace parcagpu

#endif // COLAGPU_SAMPLING_H_