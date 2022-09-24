from datasets import (
    load_data_cifar10,
    load_data_google_speech,
    load_har,
)

ARM_PSTATE_LEN = 8704
# Acceleration output buffer size
# TODO: make these adjustable on runtime
OUTPUT_LEN = 256

lea_buffer_size = {
    # (4096 - 0x138 (LEASTACK) - 2 * 8 (MSP_LEA_MAC_PARAMS)) / sizeof(int16_t)
    'msp430': 1884,
    # determined by trial and error
    'msp432': 18000,
}

# intermediate_values_size should < 65536, or TI's compiler gets confused
configs = {
    'cifar10': {
        'onnx_model': 'squeezenet_cifar10',
        'scale': 2,
        'input_scale': 10,
        'num_slots': 3,
        # 1st conv output = (15*15*64)*sizeof(int16_t)
        'intermediate_values_size': 28800,
        'data_loader': load_data_cifar10,
        'n_all_samples': 10000,
        'sample_size': [3, 32, 32],
        'op_filters': 2,
    },
    'kws': {
        'onnx_model': 'KWS-DNN_S',
        'scale': 1,
        'input_scale': 120,
        'num_slots': 2,
        # 1st fc output = (1*144*2)*sizeof(int16_t), N/T_n=2
        'intermediate_values_size': 576,
        'data_loader': load_data_google_speech,
        'n_all_samples': 4890,
        'sample_size': [25, 10],  # MFCC gives 25x10 tensors
        'op_filters': 4,
    },
    'har': {
        'onnx_model': 'HAR-CNN',
        'scale': 2,
        'input_scale': 16,
        'num_slots': 2,
        # 3rd conv output = (128*18)*sizeof(int16_t), N/T_n=2 for MSP430/JAPARI/B=1
        'intermediate_values_size': 9216,
        'data_loader': load_har,
        'n_all_samples': 2947,
        'sample_size': [9, 128],
        'op_filters': 4,
    },
}

