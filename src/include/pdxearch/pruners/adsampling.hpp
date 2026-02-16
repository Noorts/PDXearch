#pragma once

#include <Eigen/Dense>
#include <queue>
#include "pdxearch/common.hpp"

#ifdef HAS_FFTW
#include <fftw3.h>
#endif

#ifndef SKM_RESTRICT
#if defined(__GNUC__) || defined(__clang__)
#define SKM_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define SKM_RESTRICT __restrict
#elif defined(__INTEL_COMPILER)
#define SKM_RESTRICT __restrict__
#else
#define SKM_RESTRICT
#endif
#endif

namespace PDX {

/******************************************************************
 * ADSampling pruner
 ******************************************************************/
template <Quantization q = F32>
class ADSamplingPruner {
	using DISTANCES_TYPE = DistanceType_t<q>;
	using VALUE_TYPE = DataType_t<q>;
	using KNNCandidate_t = KNNCandidate<q>;
	using VectorComparator_t = VectorComparator<q>;
	using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

public:
	const uint32_t num_dimensions;

	ADSamplingPruner(const uint32_t num_dimensions, const float *matrix_p)
	    : num_dimensions(num_dimensions){
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

	template <Quantization Q = q>
	DistanceType_t<Q>
	GetPruningThreshold(uint32_t,
	                    std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>> &heap,
	                    const uint32_t current_dimension_idx) const {
		float ratio = current_dimension_idx == num_dimensions ? 1 : ratios[current_dimension_idx];
		// return std::numeric_limits<DistanceType_t<Q>>::max();
		return heap.top().distance * ratio;
	}

	void PreprocessQuery(const float *SKM_RESTRICT const raw_query_embedding,
	                     float *SKM_RESTRICT const output_query_embedding) const {
		PreprocessEmbeddings(raw_query_embedding, output_query_embedding, 1);
	}

	void PreprocessEmbeddings(const float *SKM_RESTRICT const input_embeddings,
	                          float *SKM_RESTRICT const output_embeddings, const size_t num_embeddings) const {
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
		       (1.0 + pruning_aggressiveness / std::sqrt(visited_dimensions)) * (1.0 + pruning_aggressiveness / std::sqrt(visited_dimensions));
	}

	/**
	 * @brief Rotates embeddings using the rotation matrix.
	 *
	 * Transforms embeddings to a rotated space where dimensions contribute more equally
	 * to the total distance, enabling effective early termination.
	 *
	 * For DCT path: applies sign flipping followed by DCT-II transform.
	 * For matrix path: computes out = embeddings * matrix^T.
	 *
	 * @param embeddings Input embeddings (row-major, n × num_dimensions)
	 * @param out_buffer Output buffer for rotated embeddings (n × num_dimensions)
	 * @param n Number of embeddings to rotate
	 */
	void Rotate(const VALUE_TYPE *SKM_RESTRICT const embeddings, VALUE_TYPE *SKM_RESTRICT const out_buffer,
	            const size_t n) const {
		Eigen::Map<const MatrixR> embeddings_matrix(embeddings, n, num_dimensions);
		Eigen::Map<MatrixR> out(out_buffer, n, num_dimensions);
		out.noalias() = embeddings_matrix * matrix.transpose();
	}
};

} // namespace PDX
