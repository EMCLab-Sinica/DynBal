#include "cnn_common.h"
#include "data.h"
#include "data_structures.h"
#include "double_buffering.h"
#include "layers.h"

const Node* get_node(size_t i) {
    return reinterpret_cast<const Node*>(nodes_data) + i;
}

const NodeFlags* get_node_orig_flags(size_t i) {
    return reinterpret_cast<const NodeFlags*>(node_orig_flags_data) + i;
}

const Node* get_node(const ParameterInfo* param) {
    return get_node(param->parameter_info_idx - N_INPUT);
}

NodeFlags node_flags_vm[MODEL_NODES_LEN];

template<>
uint32_t nvm_addr<NodeFlags>(uint8_t copy_id, uint16_t real_node_idx) {
    return NODE_FLAGS_OFFSET + (copy_id * MODEL_NODES_LEN + real_node_idx) * sizeof(NodeFlags);
}

template<>
NodeFlags* vm_addr<NodeFlags>(uint16_t real_node_idx) {
    return node_flags_vm + real_node_idx;
}

template<>
const char* datatype_name<NodeFlags>(void) {
    return "NodeFlags";
}

NodeFlags* get_node_flags(uint16_t node_idx) {
    NodeFlags* ret = node_flags_vm + node_idx;
    if (ret->canary != 0x55) {
        get_versioned_data<NodeFlags>(node_idx);
    }
    return ret;
}

void commit_node_flags(const NodeFlags* node_flags) {
    uint16_t node_idx = node_flags - node_flags_vm;
    commit_versioned_data<NodeFlags>(node_idx);
}
