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
class Global8Quantizer : public Quantizer {
public:
	using QUANTIZED_QUERY_TYPE = QuantizedVectorType_t<q>;

	explicit Global8Quantizer(size_t num_dimensions) : Quantizer(num_dimensions) {
	}

	uint8_t MAX_VALUE = 255;

	void PrepareQuery(const float *query, const float for_base, const float scale_factor, int32_t *dim_clip_value,
	                  QUANTIZED_QUERY_TYPE *quantized_query) {
		for (size_t i = 0; i < num_dimensions; ++i) {
			// Scale factor is global in symmetric kernel
			int rounded = static_cast<int>(std::round((query[i] - for_base) * scale_factor));
			dim_clip_value[i] = rounded;
			if (rounded > MAX_VALUE || rounded < 0) {
				quantized_query[i] = 0;
			} else {
				quantized_query[i] = static_cast<uint8_t>(rounded);
			}
		}
	};
};

} // namespace PDX
