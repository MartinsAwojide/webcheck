// SPDX-FileCopyrightText: 2026 Martins Awojide
// SPDX-FileCopyrightText: Copyright © 1997-2011 by Secret Labs AB
// SPDX-FileCopyrightText: Copyright © 1995-2011 by Fredrik Lundh and contributors
// SPDX-FileCopyrightText: Copyright © 2010 by Jeffrey 'Alex' Clark and contributors
// SPDX-License-Identifier: MIT AND MIT-CMU

// Pillow's Lanczos resize (Resample.c, 8 bits per channel) on the GPU, for the gate's 640 -> 320 tile resize.
// The coefficient tables come from webcheck.inference.resample.make_plan (bounds and 22-bit weights); this file
// only runs the two integer passes, so its bytes equal Pillow's by construction and are checked on the board.
// Build (gpu_resample.py does this): nvcc -O3 -arch=native --cudart shared -shared -Xcompiler -fPIC resample.cu
// Do not build with --use_fast_math: the float output relies on IEEE division.
#include <cuda_runtime.h>
#include <stdint.h>

#define PRECISION_BITS 22
#define MAX_TILES 8

struct Tiles {
    int n;
    int x0[MAX_TILES];
    int y0[MAX_TILES];
};

__device__ __forceinline__ uint8_t clip8(int v) {
    int i = v >> PRECISION_BITS;
    return i < 0 ? 0 : (i > 255 ? 255 : (uint8_t)i);
}

// Horizontal pass: rows [first_row, first_row + rows) of every tile of the 3-channel frame -> tmp[tile][row][x][c].
__global__ void horizontal_pass(const uint8_t *frame, int frame_stride, Tiles tiles, int first_row, int rows,
                                int out_w, int ksize, const int *bounds, const int *kk, uint8_t *tmp) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z;
    if (ox >= out_w || r >= rows) return;
    const uint8_t *line = frame + (size_t)(tiles.y0[t] + first_row + r) * frame_stride + (size_t)tiles.x0[t] * 3;
    int xmin = bounds[ox * 2], xsize = bounds[ox * 2 + 1];
    const int *k = kk + (size_t)ox * ksize;
    int s0 = 1 << (PRECISION_BITS - 1), s1 = s0, s2 = s0;
    for (int x = 0; x < xsize; x++) {
        const uint8_t *p = line + (size_t)(xmin + x) * 3;
        s0 += p[0] * k[x];
        s1 += p[1] * k[x];
        s2 += p[2] * k[x];
    }
    uint8_t *o = tmp + (((size_t)t * rows + r) * out_w + ox) * 3;
    o[0] = clip8(s0);
    o[1] = clip8(s1);
    o[2] = clip8(s2);
}

// Vertical pass: tmp[tile][row][x][c] -> out[tile][y][x][c], uint8 or float32 divided by 255 (the gate input).
__global__ void vertical_pass(const uint8_t *tmp, int rows, int w, int out_h, int ksize, const int *bounds,
                              const int *kk, int out_float, void *out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z;
    if (x >= w || oy >= out_h) return;
    int ymin = bounds[oy * 2], ysize = bounds[oy * 2 + 1];
    const int *k = kk + (size_t)oy * ksize;
    const uint8_t *col = tmp + ((size_t)t * rows * w + x) * 3;
    int s0 = 1 << (PRECISION_BITS - 1), s1 = s0, s2 = s0;
    for (int y = 0; y < ysize; y++) {
        const uint8_t *p = col + (size_t)(ymin + y) * w * 3;
        s0 += p[0] * k[y];
        s1 += p[1] * k[y];
        s2 += p[2] * k[y];
    }
    size_t at = (((size_t)t * out_h + oy) * w + x) * 3;
    if (out_float) {
        float *o = (float *)out + at;
        o[0] = (float)clip8(s0) / 255.0f;
        o[1] = (float)clip8(s1) / 255.0f;
        o[2] = (float)clip8(s2) / 255.0f;
    } else {
        uint8_t *o = (uint8_t *)out + at;
        o[0] = clip8(s0);
        o[1] = clip8(s1);
        o[2] = clip8(s2);
    }
}

// Resize n tiles of a frame in two launches on `stream`. All pointers are device pointers except tile_xy
// (n pairs of x, y on the host). Returns 0 or a CUDA error code. Does not synchronise.
extern "C" int webcheck_resample_tiles(const uint8_t *frame, int frame_stride, const int *tile_xy, int n,
                                       int first_row, int rows, int out_w, int out_h, int ksize_h, int ksize_v,
                                       const int *hb, const int *hk, const int *vb, const int *vk, uint8_t *tmp,
                                       void *out, int out_float, void *stream) {
    if (n < 1 || n > MAX_TILES) return -1;
    Tiles tiles;
    tiles.n = n;
    for (int i = 0; i < n; i++) {
        tiles.x0[i] = tile_xy[2 * i];
        tiles.y0[i] = tile_xy[2 * i + 1];
    }
    cudaStream_t s = (cudaStream_t)stream;
    dim3 block(32, 8);
    dim3 grid_h((out_w + 31) / 32, (rows + 7) / 8, n);
    horizontal_pass<<<grid_h, block, 0, s>>>(frame, frame_stride, tiles, first_row, rows, out_w, ksize_h, hb, hk, tmp);
    dim3 grid_v((out_w + 31) / 32, (out_h + 7) / 8, n);
    vertical_pass<<<grid_v, block, 0, s>>>(tmp, rows, out_w, out_h, ksize_v, vb, vk, out_float, out);
    return (int)cudaGetLastError();
}
