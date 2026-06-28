// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory>
#include <vector>

#include "encoder/base/printf_macros.h"
#include "encoder/enc_file.h"
#include "encoder/image.h"
#include "encoder/read_pfm.h"
#include "encoder/trace.h"

namespace {

struct TraceArgs {
  const char* file_in = nullptr;
  const char* trace_dir = nullptr;
  float distance = 1.0;
};

void PrintHelp(char* arg0) {
  fprintf(stderr,
          "Usage: %s <file in> <trace dir> [-d distance]\n\n"
          "  NOTE: <file in> is a .pfm file in linear SRGB colorspace\n",
          arg0);
}

}  // namespace

int main(int argc, char** argv) {
  TraceArgs args;
  for (int i = 1; i < argc; i++) {
    if (!strcmp("-h", argv[i]) || !strcmp("--help", argv[i])) {
      PrintHelp(argv[0]);
      return EXIT_SUCCESS;
    }
    if (argv[i][0] == '-' && argv[i][1] == 'd') {
      char* arg = argv[i][2] != '\0' ? &argv[i][2] : argv[++i];
      if (i == argc) {
        fprintf(stderr, "-d requires an argument\n");
        return EXIT_FAILURE;
      }
      char* end;
      args.distance = static_cast<float>(strtod(arg, &end));
      if (*end != '\0') {
        fprintf(stderr, "Unable to interpret as float: %s\n", arg);
        return EXIT_FAILURE;
      }
      continue;
    }
    if (!args.file_in) {
      args.file_in = argv[i];
    } else if (!args.trace_dir) {
      args.trace_dir = argv[i];
    }
  }
  if (!args.file_in || !args.trace_dir) {
    PrintHelp(argv[0]);
    return EXIT_FAILURE;
  }

  jxl::Image3F image;
  if (!jxl::ReadPFM(args.file_in, &image)) {
    fprintf(stderr, "Error reading PFM input file.\n");
    return EXIT_FAILURE;
  }
  fprintf(stderr, "Read %" PRIuS "x%" PRIuS " pixels input image.\n",
          image.xsize(), image.ysize());

  std::unique_ptr<jxl::EncoderTraceSink> trace =
      jxl::CreateFileTraceSink(args.trace_dir);
  std::vector<uint8_t> output;
  if (!jxl::EncodeFile(image, args.distance, &output, trace.get())) {
    fprintf(stderr, "Encoding failed.\n");
    return EXIT_FAILURE;
  }
  if (!trace->Finish()) {
    fprintf(stderr, "Failed to write trace manifest.\n");
    return EXIT_FAILURE;
  }
  fprintf(stderr, "Compressed to %" PRIuS " bytes.\n", output.size());
  fprintf(stderr, "Wrote trace artifacts to %s.\n", args.trace_dir);

  return EXIT_SUCCESS;
}
