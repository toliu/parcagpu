// Copyright 2026 The Parca Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef COLAGPU_ENV_CONFIG_H_
#define COLAGPU_ENV_CONFIG_H_

namespace parcagpu {

// Scan the process environment for all COLAGPU_* variables.
// Warns (via DEBUG_PRINTF and the error probe) about unrecognized names.
// Validates types and ranges for known variables; on invalid values, fires
// the error probe, prints a debug warning, and uses the default.
//
// Call once at startup (e.g. from init_debug).
void validateEnvVars();

} // namespace parcagpu

#endif // COLAGPU_ENV_CONFIG_H_
