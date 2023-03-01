#include <cstdint>
#include "dynbal.h"
#include "my_debug.h"

uint16_t convex_search(const UsageSpan* usage_span, uint8_t dim_idx, const uint16_t value_ranges[][2]) {
    uint16_t dim_lower = value_ranges[dim_idx][0],
             dim_upper = value_ranges[dim_idx][1];
    uint32_t cost_lower = 0, cost_upper = 0;
    while (true) {
        // Ternary search for unimodal functions
        if (dim_upper - dim_lower <= 2) {
            break;
        }
        uint16_t dim_test_1 = usage_span->nearest_value(dim_idx, (dim_lower * 2 + dim_upper) / 3),
                 dim_test_2 = usage_span->nearest_value(dim_idx, (dim_lower + dim_upper * 2) / 3);
        my_printf_debug("Testing dimension idx %d: %d, %d" NEWLINE, dim_idx, dim_test_1, dim_test_2);
        uint32_t cost_test_1 = usage_span->calc(dim_idx, dim_test_1),
                 cost_test_2 = usage_span->calc(dim_idx, dim_test_2);
        if (cost_test_1 < cost_test_2) {
            dim_upper = dim_test_1;
            cost_upper = cost_test_1;
        } else if (cost_test_1 > cost_test_2) {
            dim_lower = dim_test_2;
            cost_lower = cost_test_2;
        } else {
            dim_lower = dim_test_1;
            cost_upper = cost_test_1;
            dim_upper = dim_test_2;
            cost_upper = cost_test_2;
        }
    }
    if (!cost_lower) {
        cost_lower = usage_span->calc(dim_idx, dim_lower);
    }
    if (!cost_upper) {
        cost_upper = usage_span->calc(dim_idx, dim_upper);
    }
    if (cost_lower <= cost_upper) {
        return dim_lower;
    } else {
        return dim_upper;
    }
}
