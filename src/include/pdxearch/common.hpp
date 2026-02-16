#pragma once

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <queue>

namespace PDX {

static constexpr float PROPORTION_HORIZONTAL_DIM = 0.75f;
static constexpr size_t D_THRESHOLD_FOR_DCT_ROTATION = 512;
static constexpr size_t PDX_MAX_DIMS = 65536;
static constexpr size_t H_DIM_SIZE = 64;
static constexpr uint32_t DIMENSIONS_FETCHING_SIZES[20] = {16,  16,  32,  32,  32,  32,  64,  64,   64,   64,
                                                           128, 128, 128, 128, 256, 256, 512, 1024, 2048, 65536};

// Epsilon0 parameter of ADSampling (Reference: https://dl.acm.org/doi/abs/10.1145/3589282)
static constexpr float ADSAMPLING_PRUNING_AGGRESIVENESS = 1.5f;

template <class T, T val = 8>
static constexpr uint32_t AlignValue(T n) {
	return ((n + (val - 1)) / val) * val;
}

enum class DistanceMetric { L2SQ, COSINE, IP };

enum Quantization { F32, U8, F16, BF };

// TODO: Do the same for indexes?
template <Quantization q>
struct DistanceType {
	using type = uint32_t; // default for U8, U6, U4
};
template <>
struct DistanceType<F32> {
	using type = float;
};
template <Quantization q>
using DistanceType_t = typename DistanceType<q>::type;

// TODO: Do the same for indexes?
template <Quantization q>
struct DataType {
	using type = uint8_t; // default for U8, U6, U4
};
template <>
struct DataType<F32> {
	using type = float;
};
template <Quantization q>
using DataType_t = typename DataType<q>::type;

template <Quantization q>
struct QuantizedVectorType {
	using type = uint8_t; // default for U8, U6, U4
};
template <>
struct QuantizedVectorType<F32> {
	using type = float;
};
template <Quantization q>
using QuantizedVectorType_t = typename QuantizedVectorType<q>::type;

template <PDX::Quantization q>
struct KNNCandidate {
	uint32_t index;
	float distance;
};

template <PDX::Quantization q>
struct VectorComparator {
	bool operator()(const KNNCandidate<q> &a, const KNNCandidate<q> &b) {
		return a.distance < b.distance;
	}
};

template <Quantization q>
struct Cluster { // default for U8, U6, U4
	uint32_t num_embeddings {};
	uint32_t *indices = nullptr;
	uint8_t *data = nullptr;
};

template <>
struct Cluster<F32> {
	Cluster(uint32_t num_embeddings, uint32_t num_dimensions)
	    : num_embeddings(num_embeddings), indices(new uint32_t[num_embeddings]),
	      data(new float[static_cast<uint64_t>(num_embeddings * num_dimensions)]) {
	}

	~Cluster() {
		delete[] data;
		delete[] indices;
	}

	uint32_t num_embeddings {};
	uint32_t *indices = nullptr;
	float *data = nullptr;
};

template <Quantization q>
using Heap = typename std::priority_queue<KNNCandidate<q>, std::vector<KNNCandidate<q>>, VectorComparator<q>>;

struct PDXDimensionSplit {
	const uint32_t horizontal_dimensions;
	const uint32_t vertical_dimensions;
};

[[nodiscard]] static inline constexpr PDXDimensionSplit GetPDXDimensionSplit(const uint32_t num_dimensions) {
	auto local_proportion_horizontal_dim = PROPORTION_HORIZONTAL_DIM;
	if (num_dimensions <= 128) {
		local_proportion_horizontal_dim = 0.25;
	}
	auto horizontal_d = static_cast<uint32_t>(static_cast<float>(num_dimensions) * local_proportion_horizontal_dim);
	auto vertical_d = static_cast<uint32_t>(num_dimensions - horizontal_d);
	if (horizontal_d % H_DIM_SIZE > 0) {
		horizontal_d = ((horizontal_d + H_DIM_SIZE / 2) / H_DIM_SIZE) * H_DIM_SIZE;
		vertical_d = num_dimensions - horizontal_d;
	}
	if (!vertical_d) {
		horizontal_d = H_DIM_SIZE;
		vertical_d = num_dimensions - horizontal_d;
	}
	if (num_dimensions <= H_DIM_SIZE) {
		horizontal_d = 0;
		vertical_d = num_dimensions;
	}

	assert(horizontal_d + vertical_d == num_dimensions);

	return {horizontal_d, vertical_d};
};

static_assert(GetPDXDimensionSplit(4).horizontal_dimensions == 0);
static_assert(GetPDXDimensionSplit(4).vertical_dimensions == 4);

static_assert(GetPDXDimensionSplit(33).horizontal_dimensions == 0);
static_assert(GetPDXDimensionSplit(33).vertical_dimensions == 33);

static_assert(GetPDXDimensionSplit(64).horizontal_dimensions == 0);
static_assert(GetPDXDimensionSplit(64).vertical_dimensions == 64);

static_assert(GetPDXDimensionSplit(65).horizontal_dimensions == 0);
static_assert(GetPDXDimensionSplit(65).vertical_dimensions == 65);

static_assert(GetPDXDimensionSplit(100).horizontal_dimensions == 0);
static_assert(GetPDXDimensionSplit(100).vertical_dimensions == 100);

static_assert(GetPDXDimensionSplit(127).horizontal_dimensions == 0);
static_assert(GetPDXDimensionSplit(127).vertical_dimensions == 127);

static_assert(GetPDXDimensionSplit(128).horizontal_dimensions == 64);
static_assert(GetPDXDimensionSplit(128).vertical_dimensions == 64);

static_assert(GetPDXDimensionSplit(256).horizontal_dimensions == 192);
static_assert(GetPDXDimensionSplit(256).vertical_dimensions == 64);

static_assert(GetPDXDimensionSplit(1024).horizontal_dimensions == 768);
static_assert(GetPDXDimensionSplit(1024).vertical_dimensions == 256);

static_assert(GetPDXDimensionSplit(1028).horizontal_dimensions == 768);
static_assert(GetPDXDimensionSplit(1028).vertical_dimensions == 260);

} // namespace PDX
