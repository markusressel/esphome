#pragma once

#include "esphome/core/color.h"
#include "esphome/core/hal.h"
#include <cmath>

namespace esphome::light {

/// Binary search a monotonically increasing uint16[256] PROGMEM table.
/// Returns the largest index where table[index] <= target.
inline uint8_t gamma_table_reverse_search(const uint16_t *table, uint16_t target) {
  uint8_t lo = 0, hi = 255;
  while (lo < hi) {
    uint8_t mid = (lo + hi + 1) / 2;
    if (progmem_read_uint16(&table[mid]) <= target) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

class ESPColorCorrection {
 public:
  void set_max_brightness(const Color &max_brightness) { this->max_brightness_ = max_brightness; }
  const Color &get_max_brightness() const { return this->max_brightness_; }
  void set_local_brightness(uint8_t local_brightness) { this->local_brightness_ = local_brightness; }
  void set_gamma_table(const uint16_t *table) { this->gamma_table_ = table; }
  
  void calculate_gamma_table16(float gamma) {
    if (gamma == 0.0f) {
      for (int i = 0; i < 256; i++) {
        this->gamma_table16_[i] = (uint16_t)((i << 8) | i);
      }
    } else {
      for (int i = 0; i < 256; i++) {
        float v = (float)i / 255.0f;
        this->gamma_table16_[i] = (uint16_t)(powf(v, gamma) * 65535.0f + 0.5f);
      }
    }
  }

  void calculate_color_scales(float gamma, float master_brightness) {
      float max_r = (float)this->max_brightness_.red / 255.0f;
      float max_g = (float)this->max_brightness_.green / 255.0f;
      float max_b = (float)this->max_brightness_.blue / 255.0f;
      float max_w = (float)this->max_brightness_.white / 255.0f;
      
      float lr = max_r * master_brightness;
      float lg = max_g * master_brightness;
      float lb = max_b * master_brightness;
      float lw = max_w * master_brightness;
      
      float cr = gamma == 0.0f ? lr : powf(lr, gamma);
      float cg = gamma == 0.0f ? lg : powf(lg, gamma);
      float cb = gamma == 0.0f ? lb : powf(lb, gamma);
      float cw = gamma == 0.0f ? lw : powf(lw, gamma);
      
      this->scale_r_ = (uint32_t)(cr * 65536.0f);
      this->scale_g_ = (uint32_t)(cg * 65536.0f);
      this->scale_b_ = (uint32_t)(cb * 65536.0f);
      this->scale_w_ = (uint32_t)(cw * 65536.0f);
  }
  inline Color color_correct(Color color) const ESPHOME_ALWAYS_INLINE {
    // corrected = (uncorrected * max_brightness * local_brightness) ^ gamma
    return Color(this->color_correct_red(color.red), this->color_correct_green(color.green),
                 this->color_correct_blue(color.blue), this->color_correct_white(color.white));
  }
  inline uint8_t color_correct_red(uint8_t red) const ESPHOME_ALWAYS_INLINE {
    return this->color_correct_red_16(red) >> 8;
  }
  inline uint8_t color_correct_green(uint8_t green) const ESPHOME_ALWAYS_INLINE {
    return this->color_correct_green_16(green) >> 8;
  }
  inline uint8_t color_correct_blue(uint8_t blue) const ESPHOME_ALWAYS_INLINE {
    return this->color_correct_blue_16(blue) >> 8;
  }
  inline uint8_t color_correct_white(uint8_t white) const ESPHOME_ALWAYS_INLINE {
    return this->color_correct_white_16(white) >> 8;
  }

  inline uint16_t color_correct_red_16(uint8_t red) const ESPHOME_ALWAYS_INLINE {
    return ((uint32_t)this->gamma_table16_[red] * this->scale_r_) >> 16;
  }
  inline uint16_t color_correct_green_16(uint8_t green) const ESPHOME_ALWAYS_INLINE {
    return ((uint32_t)this->gamma_table16_[green] * this->scale_g_) >> 16;
  }
  inline uint16_t color_correct_blue_16(uint8_t blue) const ESPHOME_ALWAYS_INLINE {
    return ((uint32_t)this->gamma_table16_[blue] * this->scale_b_) >> 16;
  }
  inline uint16_t color_correct_white_16(uint8_t white) const ESPHOME_ALWAYS_INLINE {
    return ((uint32_t)this->gamma_table16_[white] * this->scale_w_) >> 16;
  }
  Color color_uncorrect(Color color) const;
  inline uint8_t color_uncorrect_red(uint8_t red) const ESPHOME_ALWAYS_INLINE {
    return this->color_uncorrect_channel_(red, this->max_brightness_.red);
  }
  inline uint8_t color_uncorrect_green(uint8_t green) const ESPHOME_ALWAYS_INLINE {
    return this->color_uncorrect_channel_(green, this->max_brightness_.green);
  }
  inline uint8_t color_uncorrect_blue(uint8_t blue) const ESPHOME_ALWAYS_INLINE {
    return this->color_uncorrect_channel_(blue, this->max_brightness_.blue);
  }
  inline uint8_t color_uncorrect_white(uint8_t white) const ESPHOME_ALWAYS_INLINE {
    return this->color_uncorrect_channel_(white, this->max_brightness_.white);
  }

 protected:
  /// Forward gamma: read uint16 PROGMEM table, convert to uint8
  uint8_t gamma_correct_(uint8_t value) const;
  /// Reverse gamma: binary search the forward PROGMEM table
  uint8_t gamma_uncorrect_(uint8_t value) const;
  /// Shared body of color_uncorrect_{red,green,blue,white}. Kept out-of-line
  /// to avoid duplicating two 16-bit divides at every call site.
  uint8_t color_uncorrect_channel_(uint8_t value, uint8_t max_brightness) const;

  const uint16_t *gamma_table_{nullptr};
  uint16_t gamma_table16_[256];
  uint32_t scale_r_{65536};
  uint32_t scale_g_{65536};
  uint32_t scale_b_{65536};
  uint32_t scale_w_{65536};
  Color max_brightness_{255, 255, 255, 255};
  uint8_t local_brightness_{255};
};

}  // namespace esphome::light
