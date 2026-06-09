> [!WARNING]
> This model is not published. Use with caution; it may not meet performance/accuracy standards and may not support some runtimes or chipsets/devices. We do not provide support for unpublished models. If this model was previously published, use earlier releases.

# [Segment-Anything-Model-3: Open-vocabulary segmentation with text and visual prompts for any concept](https://aihub.qualcomm.com/models/sam3)

SAM3 (Segment Anything with Concepts) extends SAM2 with open-vocabulary segmentation, producing bounding boxes and masks for objects matching a natural-language prompt.

This is based on the implementation of Segment-Anything-Model-3 found [here](https://github.com/facebookresearch/sam3).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/sam3).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Setup
### 1. Install the package
Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[sam3]" git+https://github.com/facebookresearch/sam3.git@f66a2514dff3c032534fe236640c1e7c0f5c7677
```

### 2. Configure Qualcomm® AI Hub Workbench
Sign-in to [Qualcomm® AI Hub Workbench](https://workbench.aihub.qualcomm.com/) with your
Qualcomm® ID. Once signed in navigate to `Account -> Settings -> API Token`.

With this API token, you can configure your client to run models on the cloud
hosted devices.
```bash
qai-hub configure --api_token API_TOKEN
```
Navigate to [docs](https://workbench.aihub.qualcomm.com/docs/) for more information.

## Run CLI Demo
Run the following simple CLI demo to verify the model is working end to end:

```bash
python -m qai_hub_models.models.sam3.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

By default, the demo will run locally in PyTorch. Pass `--eval-mode on-device` to the demo script to run the model on a cloud-hosted target device.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct. Use the following command to export the model:
```bash
python -m qai_hub_models.models.sam3.export --device "Samsung Galaxy S25 (Family)"
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of Segment-Anything-Model-3 can be found
  [here](https://github.com/facebookresearch/sam3/blob/main/LICENSE).

## References
* [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719)
* [Source Model Implementation](https://github.com/facebookresearch/sam3)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).
