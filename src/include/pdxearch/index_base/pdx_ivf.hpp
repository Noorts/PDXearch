#ifndef PDX_IVF_HPP
#define PDX_IVF_HPP

#include <cstdint>
#include <cassert>
#include <vector>
#include <memory>
#include "pdxearch/common.hpp"

namespace PDX {

template <Quantization q>
class IndexPDXIVF {};

template <>
class IndexPDXIVF<F32> {
public:
	using CLUSTER_TYPE = Cluster<F32>;

	const uint32_t num_dimensions {};
	const uint64_t total_num_embeddings {};
	const uint32_t num_clusters {};
	const uint32_t num_vertical_dimensions {};
	const uint32_t num_horizontal_dimensions {};
	std::vector<CLUSTER_TYPE> clusters;
	const bool is_ivf {};
	const bool is_normalized {};
	std::unique_ptr<float[]> centroids {};

	IndexPDXIVF(uint32_t num_dimensions, uint64_t total_num_embeddings, uint32_t num_clusters, bool is_normalized)
	    : num_dimensions(num_dimensions), total_num_embeddings(total_num_embeddings), num_clusters(num_clusters),
	      num_vertical_dimensions(GetPDXDimensionSplit(num_dimensions).vertical_dimensions),
	      num_horizontal_dimensions(GetPDXDimensionSplit(num_dimensions).horizontal_dimensions), is_ivf(true),
	      is_normalized(is_normalized) {
		clusters.reserve(num_clusters);
	}
};

} // namespace PDX

#endif // PDX_IVF_HPP
