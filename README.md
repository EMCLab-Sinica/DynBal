# Keep in Balance: Runtime-reconfigurable Intermittent Deep Inference

<!-- ABOUT THE PROJECT -->
## Overview

This project develops the DynBal middleware for runtime inference engine reconfiguration, which maintains a balance between data reuse and data refetch costs during intermittent DNN inference with respect to changing level of intermittency. Our runtime middleware includes an indirect metric to easily evaluate an inference configuration considering the variability in intermittency and a lightweight reconfiguration algorithm to efficiently optimize the configuration at runtime.

We implemented our DynBal design on the Texas Instruments device MSP-EXP432P401R. It is an ARM-based 32-bit MCU with 64KB SRAM and single instruction multiple data (SIMD) instructions for accelerated computation, and an external NVM module (Cypress CY15B104Q serial FRAM) was integrated.

Our middleware is built on top of the Stateful intermittent inference engine, and contains two key design components which interacts with the inference engine at runtime.


* Performance estimation: an indirect metric, namely _Usage Span_, for evaluating an inference configuration.
* Runtime configuration: dynamically updates the inference engine configuration parameters using feedback from performance estimation and heuristics obtained at design time.


<!-- For more technical details, please refer to our paper **TODO**. -->

Demo video: TODO

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Directory/File Structure](#directory/file-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Setup and Build](#setup-and-build)

## Directory/File Structure

Below is an explanation of the directories/files found in this repo.

* `common/conv.cpp`, `common/fc.cpp`, `common/pooling.cpp`, `common/op_handlers.cpp`, `common/op_utils.*`: functions implementing various neural network layers and auxiliary functions shared among different layers.
* `common/cnn_common.*`, `common/intermittent-cnn.*`: main components of the Stateful intermittent inference engine.
* `common/dynbal*`: functions that implement the aforementioned key design components of DynBal.
* `common/platform.*`, `common/plat-mcu.*` and `common/plat-pc.*`: high-level wrappers for handling platform-specific peripherals.
* `common/my_dsplib.*`: high-level wrappers for accessing different vendor-specific library calls performing accelerated computations.
* `common/counters.*` : helper functions for measuring runtime overhead.
* `dnn-models/`: pre-trained models and python scripts for model training, converting different model formats to ONNX and converting a model into a custom format recognized by the lightweight inference engine.
* `dnn-models/dynbal.py`: functions for offline analysis of the Usage Span metric.
* `msp432/`: platform-speicific hardware initialization functions.
* `tools/`: helper functions for various system peripherals (e.g., UART, system clocks and external FRAM).

## Getting Started

### Prerequisites

Here are basic software and hardware requirements to build DynBal along with the Stateful intermittent inference engine:

* Python 3.9
* Several deep learning Python libraries defined in `requirements.txt`. Those libraries can be installed with `pip3 install -r requirements.txt`.
* [Code composer studio](https://www.ti.com/tool/CCSTUDIO) 12.0
* [MSP-EXP432P401R LaunchPad](https://www.ti.com/tool/MSP-EXP432P401R)
* [MSP432 driverlib](https://www.ti.com/tool/MSPDRIVERLIB) 3.21.00.05

### Setup and Build

1. Prepare vendor-supplied libraries for hardware-accelerated computation. `git submodule update --init --recursive` will download them all.
1. Convert the provided pre-trained models with the command `python3 dnn-models/transform.py --target msp432 --stateful (cifar10|har|kws)` to specify the model to deploy from one of `cifar10`, `har` or `kws`.
1. Download and extract MSP432 driverlib, and copy `driverlib/MSP432P4xx` folder into the `msp432/` folder.
1. Import the folder `msp432/` as a project in CCStudio.
