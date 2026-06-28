// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_TRACE_H_
#define ENCODER_TRACE_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "encoder/ac_strategy.h"
#include "encoder/base/status.h"
#include "encoder/image.h"

namespace jxl {

struct TraceImageInfo {
  size_t xsize = 0;
  size_t ysize = 0;
  float distance = 0.0f;
};

class EncoderTraceSink {
 public:
  virtual ~EncoderTraceSink() {}

  virtual Status BeginImage(const TraceImageInfo& image_info) = 0;

  virtual Status WriteBytes(const std::string& name, const uint8_t* data,
                            size_t size) = 0;

  virtual Status WriteArray(const std::string& name, const std::string& dtype,
                            const std::vector<size_t>& shape,
                            const void* data, size_t element_size,
                            const std::string& tolerance) = 0;

  virtual Status Finish() = 0;
};

std::unique_ptr<EncoderTraceSink> CreateFileTraceSink(const char* trace_dir);

Status TraceImage3F(EncoderTraceSink* trace, const std::string& name,
                    const Image3F& image, const std::string& tolerance);
Status TraceImage3S(EncoderTraceSink* trace, const std::string& name,
                    const Image3S& image);
Status TraceImageF(EncoderTraceSink* trace, const std::string& name,
                   const ImageF& image, const std::string& tolerance);
Status TraceImageFRect(EncoderTraceSink* trace, const std::string& name,
                       const ImageF& image, Rect rect,
                       const std::string& tolerance);
Status TraceImageB(EncoderTraceSink* trace, const std::string& name,
                   const ImageB& image);
Status TraceImageSB(EncoderTraceSink* trace, const std::string& name,
                    const ImageSB& image);
Status TraceImageBRect(EncoderTraceSink* trace, const std::string& name,
                       const ImageB& image, Rect rect);
Status TraceAcStrategy(EncoderTraceSink* trace, const std::string& name,
                       const AcStrategyImage& ac_strategy);

}  // namespace jxl

#endif  // ENCODER_TRACE_H_
