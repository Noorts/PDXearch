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
template <Quantization q = F32>
class ADSamplingPruner {
	using DISTANCES_TYPE = DistanceType_t<q>;
	using VALUE_TYPE = DataType_t<q>;
	using KNNCandidate_t = KNNCandidate<F32>;
	using VectorComparator_t = VectorComparator<F32>;
	using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

public:
	const uint32_t num_dimensions;

	ADSamplingPruner(const uint32_t num_dimensions, const float *matrix_p) : num_dimensions(num_dimensions) {
		ratios.resize(num_dimensions);
		for (size_t i = 0; i < num_dimensions; ++i) {
			ratios[i] = GetRatio(i);
		}
#ifdef HAS_FFTW
		if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION) {
			matrix = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
			    matrix_p, 1, num_dimensions);
		} else {
			matrix = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
			    matrix_p, num_dimensions, num_dimensions);
		}
#else
		matrix = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
		    matrix_p, num_dimensions, num_dimensions);
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

	float GetPruningThreshold(
	    uint32_t,
	    std::priority_queue<KNNCandidate<F32>, std::vector<KNNCandidate<F32>>, VectorComparator<F32>> &heap,
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
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix;
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
		Eigen::Map<const MatrixR> embeddings_matrix(embeddings, n, num_dimensions);
		Eigen::Map<MatrixR> out(out_buffer, n, num_dimensions);
		out.noalias() = embeddings_matrix * matrix.transpose();
	}
};

} // namespace PDX
