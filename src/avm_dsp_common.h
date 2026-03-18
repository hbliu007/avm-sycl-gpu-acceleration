// avm_dsp/avm_dsp_common.h
/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * Common definitions for AVM DSP operations.
 */

#ifndef AVM_DSP_COMMON_H_
#define AVM_DSP_COMMON_H_

#include <cstdint>

// Common type definitions
typedef int32_t tran_low_t;
typedef int16_t tran_high_t;

// Block size constants
#define BLOCK_SIZE_8 8
#define BLOCK_SIZE_16 16
#define BLOCK_SIZE_32 32
#define BLOCK_SIZE_64 64

// Pixel type
typedef uint8_t aom_pixel_t;

// Bit depth (default 8-bit)
#ifndef AOM_BITS
#define AOM_BITS 8
#endif

#endif  // AVM_DSP_COMMON_H_
