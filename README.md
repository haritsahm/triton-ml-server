# Triton Machine Learning Model Collections

This is part of [cpp-ml-server](https://github.com/haritsahm/cpp-ml-server) project that holds the models configurations for Triton Inference Engine.

## Build Model

Please read how to use different model sources and model configurations with [Triton Inference Server Guide](https://github.com/triton-inference-server/server/tree/main/docs#user-guide)

1. Load and export Timm Classification Model to ONNX
```
// Modify the model name in the script then run
python3 export_onnx.py
```

## Model Repository
1. Imagenet Classification Static
   - Name: `imagenet_classification_static`
   - Max batch size: 1
   - Model origin: `timm/efficientnetv2_rw_s`