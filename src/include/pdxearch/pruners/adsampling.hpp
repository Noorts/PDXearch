#pragma once

#include <Eigen/Dense>
#include <queue>
#include "pdxearch/common.hpp"

#ifdef HAS_FFTW
#include <fftw3.h>
#endif

namespace PDX {

/******************************************************************
 * ADSampling pruner
 ******************************************************************/
template <Quantization Q = F32>
class ADSamplingPruner {
	using matrix_t = eigen_matrix_t;

public:
	const uint32_t num_dimensions;

	ADSamplingPruner(const uint32_t num_dimensions, const float *matrix_p) : num_dimensions(num_dimensions) {
		ratios.resize(num_dimensions);
		for (size_t i = 0; i < num_dimensions; ++i) {
			ratios[i] = GetRatio(i);
		}
#ifdef HAS_FFTW
		if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION) {
			matrix = Eigen::Map<const matrix_t>(matrix_p, 1, num_dimensions);
		} else {
			matrix = Eigen::Map<const matrix_t>(matrix_p, num_dimensions, num_dimensions);
		}
#else
		matrix = Eigen::Map<const matrix_t>(matrix_p, num_dimensions, num_dimensions);
#endif
	}

	void SetPruningAggresiveness(const float pruning_aggressiveness) {
		ADSamplingPruner::pruning_aggressiveness = pruning_aggressiveness;
		for (size_t i = 0; i < num_dimensions; ++i) {
			ratios[i] = GetRatio(i);
		}
	}

	void SetMatrix(const Eigen::MatrixXf &matrix) {
		ADSamplingPruner::matrix = matrix;
	}

	float GetPruningThreshold(uint32_t,
	                          std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> &heap,
	                          const uint32_t current_dimension_idx) const {
		float ratio = current_dimension_idx == num_dimensions ? 1 : ratios[current_dimension_idx];
		return heap.top().distance * ratio;
	}

	void PreprocessQuery(const float *PDX_RESTRICT const raw_query_embedding,
	                     float *PDX_RESTRICT const output_query_embedding) const {
		PreprocessEmbeddings(raw_query_embedding, output_query_embedding, 1);
	}

	void PreprocessEmbeddings(const float *PDX_RESTRICT const input_embeddings,
	                          float *PDX_RESTRICT const output_embeddings, const size_t num_embeddings) const {
		Rotate(input_embeddings, output_embeddings, num_embeddings);
	}

private:
	float pruning_aggressiveness = ADSAMPLING_PRUNING_AGGRESIVENESS;
	matrix_t matrix;
	std::vector<float> ratios;

	float GetRatio(const size_t &visited_dimensions) const {
		if (visited_dimensions == 0) {
			return 1;
		}
		if (visited_dimensions == num_dimensions) {
			return 1.0;
		}
		return static_cast<float>(visited_dimensions) / num_dimensions *
		       (1.0 + pruning_aggressiveness / std::sqrt(visited_dimensions)) *
		       (1.0 + pruning_aggressiveness / std::sqrt(visited_dimensions));
	}

	/**
	 * @brief Rotates embeddings using the rotation matrix.
	 *
	 * Transforms embeddings to a rotated space where dimensions contribute more equally
	 * to the total distance, enabling effective early termination.
	 *
	 * @param embeddings Input embeddings (row-major, n × num_dimensions)
	 * @param out_buffer Output buffer for rotated embeddings (n × num_dimensions)
	 * @param n Number of embeddings to rotate
	 */
	void Rotate(const float *PDX_RESTRICT const embeddings, float *PDX_RESTRICT const out_buffer,
	            const size_t n) const {
		const char trans_a = 'T';
		const char trans_b = 'N';
		const float alpha = 1.0f;
		const float beta = 0.0f;
		int dim = static_cast<int>(num_dimensions);
		int n_blas = static_cast<int>(n);
		sgemm_(&trans_a, &trans_b, &dim, &n_blas, &dim, &alpha, matrix.data(), &dim, embeddings, &dim, &beta,
		       out_buffer, &dim);
	}
};

} // namespace PDX
