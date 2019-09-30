// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// A simple main function to run a MediaPipe graph.

#include "absl/strings/str_split.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/calculators/image/opencv_image_encoder_calculator.pb.h"

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");

DEFINE_string(input_image, "", "Path to input image");
DEFINE_string(output_image, "", "Path to output image");

::mediapipe::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
    mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
      calculator_graph_config_contents);
  std::string input_image_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
    FLAGS_input_image, &input_image_contents));
  LOG(INFO) << "Load input image";

  ::mediapipe::Packet input_packet = ::mediapipe::MakePacket<std::string>(input_image_contents)
    .At(mediapipe::Timestamp(0));

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller("encoded_output_image"));

  LOG(INFO) << "Start running the calculator graph.";
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "encoded_input_image", input_packet));

  MP_RETURN_IF_ERROR(graph.CloseInputStream("encoded_input_image"));
  mediapipe::Packet packet;
  // Get the output packets std::string.
  poller.Next(&packet);
  auto result = packet.Get<mediapipe::OpenCvImageEncoderCalculatorResults>();

  MP_RETURN_IF_ERROR(mediapipe::file::SetContents(
      FLAGS_output_image, result.encoded_image()));

  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  auto run_status = RunMPPGraph();
  LOG(INFO) << run_status.message() << run_status << run_status.error_message();
  CHECK(run_status.ok());
  return 0;
}