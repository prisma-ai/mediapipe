# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//mediapipe/examples:__subpackages__"])


cc_library(
    name = "face_detection_deps",
    deps = [
        "//mediapipe/calculators/image:opencv_encoded_image_to_image_frame_calculator",
        "//mediapipe/calculators/image:opencv_image_encoder_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detection_letterbox_removal_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
    ],
    data = [
        "//mediapipe/models:face_detection_front.tflite",
        "//mediapipe/models:face_detection_front_labelmap.txt",
    ],
)

cc_binary(
    name = "face_detection",
    srcs = ["run_face_detection.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
        ":face_detection_deps",
    ],

)
