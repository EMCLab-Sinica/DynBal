from datasets import (
    load_data_cifar10,
    load_data_google_speech,
    load_har,
)

# intermediate_values_size should < 65536, or TI's compiler gets confused
configs = {
    'cifar10': {
        'onnx_model': 'squeezenet_cifar10',
        'scale': 2,
        'input_scale': 10,
        'num_slots': 3,
        'intermediate_values_size': 65000,
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
        'intermediate_values_size': 20000,
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
        'intermediate_values_size': 20000,
        'data_loader': load_har,
        'n_all_samples': 2947,
        'sample_size': [9, 128],
        'op_filters': 4,
    },
}

