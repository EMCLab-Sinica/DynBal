from datasets import (
    load_data_cifar10,
    load_data_google_speech,
    load_har,
)

ARM_PSTATE_LEN = None
# Acceleration output buffer size
# TODO: make these adjustable on runtime
OUTPUT_LEN = 256

vm_size = {
    # (4096 - 0x138 (LEASTACK) - 2 * 8 (MSP_LEA_MAC_PARAMS)) / sizeof(int16_t)
    'msp430': 1884,
    # determined by trial and error
    'msp432': 26704,  # includes space for pState
}

configs = {
    'cifar10': {
        'onnx_model': 'squeezenet_cifar10',
        'scale': 2,
        'input_scale': 10,
        'num_slots': 3,
        'data_loader': load_data_cifar10,
        'n_all_samples': 10000,
        'op_filters': 2,
    },
    'cifar10-cnn': {
        'onnx_model': 'cifar10-cnn',
        'scale': 2,
        'input_scale': 10,
        'num_slots': 2,
        'data_loader': load_data_cifar10,
        'n_all_samples': 10000,
        'op_filters': 2,
    },
    'kws': {
        'onnx_model': 'KWS-DNN_S',
        'scale': 1,
        'input_scale': 120,
        'num_slots': 2,
        'data_loader': load_data_google_speech,
        'n_all_samples': 4890,
        'op_filters': 4,
    },
    'har': {
        'onnx_model': 'HAR-CNN',
        'scale': 2,
        'input_scale': 16,
        'num_slots': 2,
        'data_loader': load_har,
        'n_all_samples': 2947,
        'op_filters': 4,
    },
}

