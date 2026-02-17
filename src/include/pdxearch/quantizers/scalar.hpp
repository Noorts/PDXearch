#pragma once

#include <cstdint>
#include <cmath>
#include "pdxearch/common.hpp"

namespace PDX {

class Quantizer {

public:
	explicit Quantizer(size_t num_dimensions) : num_dimensions(num_dimensions) {
	}
	virtual ~Quantizer() = default;

public:
	void NormalizeQuery(const float *src, float *out) const {
		float sum = 0.0f;
		for (size_t i = 0; i < num_dimensions; ++i) {
			sum += src[i] * src[i];
		}

		if (sum == 0.0f) {
			return;
		}

		float norm = std::sqrt(sum);
		for (size_t i = 0; i < num_dimensions; ++i) {
			out[i] = src[i] / norm;
		}
	}
	const size_t num_dimensions;
};

template <Quantization q = U8>
class ScalarQuantizer : public Quantizer {
public:
	using quantized_query_t = QuantizedVectorType_t<q>;

	explicit ScalarQuantizer(size_t num_dimensions) : Quantizer(num_dimensions) {
	}

	uint8_t MAX_VALUE = 255;

	void PrepareQuery(const float *query, const float quantization_base, const float quantization_scale,
		quantized_query_t *quantized_query) {
		for (size_t i = 0; i < num_dimensions; ++i) {
			int rounded = static_cast<int>(std::round((query[i] - quantization_base) * quantization_scale));
			if (PDX_UNLIKELY(rounded > MAX_VALUE)) {
				quantized_query[i] = MAX_VALUE;
			} else if (PDX_UNLIKELY(rounded < 0)) {
				quantized_query[i] = 0;
			} else {
				quantized_query[i] = static_cast<uint8_t>(rounded);
			}
		}
	};
};

} // namespace PDX
