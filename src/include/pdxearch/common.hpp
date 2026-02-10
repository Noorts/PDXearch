#ifndef PDX_COMMON_HPP
#define PDX_COMMON_HPP

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <queue>

namespace PDX {

static constexpr float PROPORTION_VERTICAL_DIM = 0.25;
static constexpr size_t D_THRESHOLD_FOR_DCT_ROTATION = 512;
static constexpr size_t PDX_VECTOR_SIZE = 64;
static constexpr size_t PDX_MAX_DIMS = 4096;
static constexpr size_t PDX_MIN_DIMS = 128;
static constexpr size_t MAX_EMBEDDINGS_PER_CLUSTER = 10240;

template <class T, T val = 8>
static constexpr uint32_t AlignValue(T n) {
	return ((n + (val - 1)) / val) * val;
}

enum DimensionsOrder {
	SEQUENTIAL,
	DISTANCE_TO_MEANS,
	DECREASING,
	DISTANCE_TO_MEANS_IMPROVED,
	DECREASING_IMPROVED,
	DIMENSION_ZONES
};

// Distance metric functionality that is exposed externally.
enum class DistanceMetric { L2SQ, COSINE, IP };

// Used internally.
enum DistanceFunction {
	L2,
	IP,
	L1,
	NEGATIVE_L2 // Only the negative term of L2 (-2*q[i]*d[i])
};

enum Quantization {
	F32,
	U8,
	// TODO:
	F16,
	BF,
	U6,
	U4,
	ASYMMETRIC_U8,
	ASYMMETRIC_LEP_U8
};

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
	// Based on the original PDX code (see link) but with PROPORTION_VERTICAL_DIM fixed (in the original code the
	// constant represents the horizontal, not the vertical portion):
	// https://github.com/cwida/PDX/blob/4a2e65e90e177155c42d277b5523c4fd2fa35540/python/pdxearch/index_base.py#L119-L127

	assert(num_dimensions % 4 == 0);
	assert(num_dimensions >= PDX_MIN_DIMS);
	assert(num_dimensions <= PDX_MAX_DIMS);

	// Special case for 128 dimensions, to avoid {128, 0} split.
	if (num_dimensions == 128) {
		return {64, 64};
	}

	uint32_t v_dims = static_cast<uint32_t>(static_cast<float>(num_dimensions) * PDX::PROPORTION_VERTICAL_DIM);
	uint32_t h_dims = num_dimensions - v_dims;

	// Round the horizontal dimensions to the nearest multiple of PDX_VECTOR_SIZE.
	if (h_dims % PDX_VECTOR_SIZE != 0) {
		h_dims = ((h_dims + PDX_VECTOR_SIZE / 2) / PDX_VECTOR_SIZE) * PDX_VECTOR_SIZE;
		v_dims = num_dimensions - h_dims;
	}

	return {h_dims, v_dims};
};

static_assert(GetPDXDimensionSplit(128).horizontal_dimensions == 64);
static_assert(GetPDXDimensionSplit(128).vertical_dimensions == 64);

static_assert(GetPDXDimensionSplit(256).horizontal_dimensions == 192);
static_assert(GetPDXDimensionSplit(256).vertical_dimensions == 64);

static_assert(GetPDXDimensionSplit(1024).horizontal_dimensions == 768);
static_assert(GetPDXDimensionSplit(1024).vertical_dimensions == 256);

static_assert(GetPDXDimensionSplit(1028).horizontal_dimensions == 768);
static_assert(GetPDXDimensionSplit(1028).vertical_dimensions == 260);

}; // namespace PDX

#endif // PDX_COMMON_HPP
