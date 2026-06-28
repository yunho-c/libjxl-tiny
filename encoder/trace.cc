// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/trace.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include <sstream>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "encoder/config.h"

namespace jxl {
namespace {

struct TraceArtifact {
  std::string name;
  std::string path;
  std::string dtype;
  std::vector<size_t> shape;
  std::string tolerance;
  size_t bytes = 0;
};

std::string JoinPath(const std::string& dir, const std::string& basename) {
  if (dir.empty()) return basename;
  const char last = dir[dir.size() - 1];
  if (last == '/' || last == '\\') return dir + basename;
  return dir + "/" + basename;
}

Status MakeDirIfNeeded(const std::string& dir) {
#if defined(_WIN32)
  if (_mkdir(dir.c_str()) == 0 || errno == EEXIST) return true;
#else
  if (mkdir(dir.c_str(), 0777) == 0 || errno == EEXIST) return true;
#endif
  return JXL_FAILURE("Failed to create trace directory %s: %s", dir.c_str(),
                     strerror(errno));
}

Status WriteFile(const std::string& path, const uint8_t* data, size_t size) {
  FILE* file = fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return JXL_FAILURE("Failed to open trace artifact %s: %s", path.c_str(),
                       strerror(errno));
  }
  if (size != 0 && fwrite(data, 1, size, file) != size) {
    const int saved_errno = errno;
    fclose(file);
    return JXL_FAILURE("Failed to write trace artifact %s: %s", path.c_str(),
                       strerror(saved_errno));
  }
  if (fclose(file) != 0) {
    return JXL_FAILURE("Failed to close trace artifact %s: %s", path.c_str(),
                       strerror(errno));
  }
  return true;
}

std::string JsonEscape(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  for (char c : in) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
  }
  return out;
}

std::string NpyDescr(const std::string& dtype) {
  if (dtype == "float32") return "<f4";
  if (dtype == "float64") return "<f8";
  if (dtype == "uint8") return "|u1";
  if (dtype == "int8") return "|i1";
  if (dtype == "uint16") return "<u2";
  if (dtype == "int16") return "<i2";
  if (dtype == "uint32") return "<u4";
  if (dtype == "int32") return "<i4";
  return "";
}

std::string NpyShape(const std::vector<size_t>& shape) {
  std::ostringstream out;
  out << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out << ", ";
    out << shape[i];
  }
  if (shape.size() == 1) out << ",";
  out << ")";
  return out.str();
}

Status WriteNpy(const std::string& path, const std::string& dtype,
                const std::vector<size_t>& shape, const void* data,
                size_t element_size) {
  const std::string descr = NpyDescr(dtype);
  if (descr.empty()) {
    return JXL_FAILURE("Unsupported trace array dtype: %s", dtype.c_str());
  }
  size_t num_elements = 1;
  for (size_t dim : shape) {
    num_elements *= dim;
  }
  std::string header = "{'descr': '" + descr +
                       "', 'fortran_order': False, 'shape': " +
                       NpyShape(shape) + ", }";
  const size_t preamble_size = 10;
  const size_t padding =
      (16 - ((preamble_size + header.size() + 1) % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');
  if (header.size() > 0xFFFFu) {
    return JXL_FAILURE("Trace npy header too large");
  }

  FILE* file = fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return JXL_FAILURE("Failed to open trace array %s: %s", path.c_str(),
                       strerror(errno));
  }
  const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  const uint16_t header_size = static_cast<uint16_t>(header.size());
  const uint8_t header_size_le[] = {
      static_cast<uint8_t>(header_size & 0xFF),
      static_cast<uint8_t>((header_size >> 8) & 0xFF)};
  bool ok = fwrite(magic, 1, sizeof(magic), file) == sizeof(magic) &&
            fwrite(header_size_le, 1, sizeof(header_size_le), file) ==
                sizeof(header_size_le) &&
            fwrite(header.data(), 1, header.size(), file) == header.size();
  const size_t byte_size = num_elements * element_size;
  if (ok && byte_size != 0) {
    ok = fwrite(data, 1, byte_size, file) == byte_size;
  }
  if (!ok) {
    const int saved_errno = errno;
    fclose(file);
    return JXL_FAILURE("Failed to write trace array %s: %s", path.c_str(),
                       strerror(saved_errno));
  }
  if (fclose(file) != 0) {
    return JXL_FAILURE("Failed to close trace array %s: %s", path.c_str(),
                       strerror(errno));
  }
  return true;
}

class FileTraceSink : public EncoderTraceSink {
 public:
  explicit FileTraceSink(const char* trace_dir) : trace_dir_(trace_dir) {}

  Status BeginImage(const TraceImageInfo& image_info) override {
    image_info_ = image_info;
    JXL_RETURN_IF_ERROR(MakeDirIfNeeded(trace_dir_));
    return true;
  }

  Status WriteBytes(const std::string& name, const uint8_t* data,
                    size_t size) override {
    const std::string path = name + ".bin";
    JXL_RETURN_IF_ERROR(WriteFile(JoinPath(trace_dir_, path), data, size));
    TraceArtifact artifact;
    artifact.name = name;
    artifact.path = path;
    artifact.dtype = "uint8";
    artifact.shape.push_back(size);
    artifact.tolerance = "exact";
    artifact.bytes = size;
    artifacts_.push_back(artifact);
    return true;
  }

  Status WriteArray(const std::string& name, const std::string& dtype,
                    const std::vector<size_t>& shape, const void* data,
                    size_t element_size,
                    const std::string& tolerance) override {
    const std::string path = name + ".npy";
    JXL_RETURN_IF_ERROR(
        WriteNpy(JoinPath(trace_dir_, path), dtype, shape, data, element_size));
    size_t num_elements = 1;
    for (size_t dim : shape) {
      num_elements *= dim;
    }
    TraceArtifact artifact;
    artifact.name = name;
    artifact.path = path;
    artifact.dtype = dtype;
    artifact.shape = shape;
    artifact.tolerance = tolerance;
    artifact.bytes = num_elements * element_size;
    artifacts_.push_back(artifact);
    return true;
  }

  Status Finish() override {
    std::ostringstream manifest;
    manifest << "{\n";
    manifest << "  \"schema_version\": 1,\n";
    manifest << "  \"distance\": " << image_info_.distance << ",\n";
    manifest << "  \"image\": {\n";
    manifest << "    \"xsize\": " << image_info_.xsize << ",\n";
    manifest << "    \"ysize\": " << image_info_.ysize << "\n";
    manifest << "  },\n";
    manifest << "  \"build\": {\n";
    manifest << "    \"OPTIMIZE_CODE\": " << OPTIMIZE_CODE << ",\n";
    manifest << "    \"OPTIMIZE_CHROMA_FROM_LUMA\": "
             << OPTIMIZE_CHROMA_FROM_LUMA << ",\n";
    manifest << "    \"OPTIMIZE_BLOCK_SIZES\": " << OPTIMIZE_BLOCK_SIZES
             << "\n";
    manifest << "  },\n";
    manifest << "  \"artifacts\": [\n";
    for (size_t i = 0; i < artifacts_.size(); ++i) {
      const TraceArtifact& artifact = artifacts_[i];
      manifest << "    {\n";
      manifest << "      \"name\": \"" << JsonEscape(artifact.name)
               << "\",\n";
      manifest << "      \"path\": \"" << JsonEscape(artifact.path)
               << "\",\n";
      manifest << "      \"dtype\": \"" << JsonEscape(artifact.dtype)
               << "\",\n";
      manifest << "      \"shape\": [";
      for (size_t j = 0; j < artifact.shape.size(); ++j) {
        if (j != 0) manifest << ", ";
        manifest << artifact.shape[j];
      }
      manifest << "],\n";
      manifest << "      \"tolerance\": \""
               << JsonEscape(artifact.tolerance) << "\",\n";
      manifest << "      \"bytes\": " << artifact.bytes << "\n";
      manifest << "    }";
      if (i + 1 != artifacts_.size()) manifest << ",";
      manifest << "\n";
    }
    manifest << "  ]\n";
    manifest << "}\n";
    const std::string manifest_str = manifest.str();
    JXL_RETURN_IF_ERROR(
        WriteFile(JoinPath(trace_dir_, "manifest.json"),
                  reinterpret_cast<const uint8_t*>(manifest_str.data()),
                  manifest_str.size()));
    return true;
  }

 private:
  std::string trace_dir_;
  TraceImageInfo image_info_;
  std::vector<TraceArtifact> artifacts_;
};

}  // namespace

std::unique_ptr<EncoderTraceSink> CreateFileTraceSink(const char* trace_dir) {
  return std::unique_ptr<EncoderTraceSink>(new FileTraceSink(trace_dir));
}

}  // namespace jxl
