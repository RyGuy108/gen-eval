#!/usr/bin/env bash
# Download scorer weights into ./weights (not committed).
#   CLIP ViT-L/14 ONNX (OpenAI weights, exported by the clip-as-service project)
#   YuNet face detector + SFace recognizer (OpenCV model zoo)
#   YOLOE-11L-seg + MobileCLIP text encoder (adherence fact checks)
set -euo pipefail
W=${GENEVAL_WEIGHTS:-weights}
mkdir -p "$W/clip" "$W/face"
ARCH=${GENEVAL_CLIP:-ViT-L-14}
for part in visual textual; do
  f="$W/clip/$ARCH-$part.onnx"
  [ -s "$f" ] || curl -fL -o "$f" "https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/$ARCH/$part.onnx"
done
Z=https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models
[ -s "$W/face/face_detection_yunet_2023mar.onnx" ] || curl -fL -o "$W/face/face_detection_yunet_2023mar.onnx" \
  "$Z/face_detection_yunet/face_detection_yunet_2023mar.onnx"
[ -s "$W/face/face_recognition_sface_2021dec.onnx" ] || curl -fL -o "$W/face/face_recognition_sface_2021dec.onnx" \
  "$Z/face_recognition_sface/face_recognition_sface_2021dec.onnx"
# YOLOE open-vocabulary detector + its MobileCLIP text encoder (Ultralytics release assets)
mkdir -p "$W/det"
A=https://github.com/ultralytics/assets/releases/download/v8.3.0
[ -s "$W/det/yoloe-11l-seg.pt" ] || curl -fL -o "$W/det/yoloe-11l-seg.pt" "$A/yoloe-11l-seg.pt"
[ -s "$W/mobileclip_blt.ts" ] || curl -fL -o "$W/mobileclip_blt.ts" "$A/mobileclip_blt.ts"
ls -la "$W"/*
