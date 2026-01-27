#ifndef PDX_COMMON_HPP
#define PDX_COMMON_HPP

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <queue>

namespace PDX {

static inline float PROPORTION_VERTICAL_DIM = 0.25;
static inline size_t D_THRESHOLD_FOR_DCT_ROTATION = 512;
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

}; // namespace PDX

#endif // PDX_COMMON_HPP
