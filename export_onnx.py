import torch
import timm
import onnx

def main():
    pass

if __name__ == "__main__":
    model = timm.create_model('efficientnetv2_rw_s', pretrained=True)

    config = model.default_cfg
    img_size = config["test_input_size"] if "test_input_size" in config else config["input_size"]
    print(img_size)

    dummy_input = torch.randn(1, *img_size)
    model.eval()
    torch.onnx.export(model, dummy_input, "efficientnetv2_rw_s_static.onnx",
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    # 'output' : {0 : 'batch_size'}},
        )

