#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Reproduce greedy_round_robin's zero-multiplied impossible entries and
// first-index argmax; do not replace them with negative infinity.
extern "C" int fairrec_allocate(const float* scores, const uint8_t* mask, const int32_t* order,
                                 int m, int n, int k, double alpha,
                                 int* selected, int* lengths, int* remaining) {
    const int quota = int(alpha * m * k / double(n));
    const int rounds = int(std::ceil(quota * n / double(m)));
    int T = quota * n;
    std::vector<int> available(n, quota);
    std::vector<uint8_t> feasible(mask, mask + size_t(m) * n);
    std::fill(lengths, lengths + m, 0);
    for (int t = 0; t < rounds; ++t) {
        for (int u = 0; u < m; ++u) {
            if (T == 0) { *remaining = T; return quota; }
            const size_t offset = size_t(u) * n;
            int best = -1;
            // Positive values always beat the implicit zeros. Pre-sorted scores
            // retain ascending item-id tie breaking, identical to np.argmax.
            for (int j = 0; j < n; ++j) {
                int p = order[offset + j];
                if (scores[offset + p] <= 0) break;
                if (available[p] > 0 && feasible[offset + p]) { best = p; break; }
            }
            if (best < 0) {
                best = 0;
                float best_value = (available[0] > 0 && feasible[offset]) ? scores[offset] : 0.0f;
                for (int p = 1; p < n; ++p) {
                    float v = (available[p] > 0 && feasible[offset + p]) ? scores[offset + p] : 0.0f;
                    if (v > best_value) { best_value = v; best = p; }
                }
            }
            if (available[best] > 0 && feasible[offset + best]) {
                if (lengths[u] >= k) { *remaining = -1; return quota; }
                selected[size_t(u) * k + lengths[u]++] = best;
                feasible[offset + best] = 0;
                --available[best]; --T;
            } else { *remaining = T; return quota; }
        }
    }
    *remaining = T;
    return quota;
}

extern "C" int fairrec_complete(int* selected, int* lengths, const int32_t* top,
                                 int m, int k) {
    for (int u = 0; u < m; ++u) {
        for (int j = 0; j < 2*k && lengths[u] < k; ++j) {
            int p = top[size_t(u) * 2*k + j];
            bool seen = false;
            for (int q = 0; q < lengths[u]; ++q)
                if (selected[size_t(u)*k+q] == p) { seen = true; break; }
            if (!seen) selected[size_t(u)*k + lengths[u]++] = p;
        }
        if (lengths[u] != k) return 1;
    }
    return 0;
}
