provider colagpu {
  probe api_correlation(uint32_t correlationId, int signedCbid,
                          const char *name);
  /* syncKind: vendor-neutral synchronize kind, partitioned by backend.
     100-199  CUPTI (NVIDIA) driver synchronize APIs
     200-299  MSPTI (Ascend) synchronize APIs (reserved) */
  probe api_synchronize(uint64_t start, uint64_t end, uint32_t syncKind);
  probe kernel_executed(uint64_t start, uint64_t end,
                        uint32_t correlationId, uint32_t deviceId,
                        uint32_t streamId, uint32_t graphId,
                        uint64_t graphNodeId, const char *name);
  probe activity_batch(const void **ptrs, uint32_t count);
  probe error(int32_t code, const char *message, const char *component);
  
  probe pc_sample_batch(const void **records, uint32_t count);
  probe stall_reason_map(const char *names, uint32_t count);
  probe cubin_loaded(uint64_t cubinCrc, const char *cubin, uint64_t cubinSize);
  probe cubin_unloaded(uint64_t cubinCrc);
  probe gpu_config(uint32_t deviceId, uint32_t samplingFactor,
                    uint32_t clockKHz, uint32_t smCount);
};
