#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" int fa_latency_order(int n, const int32_t *a, const int32_t *b, int32_t *order)
{
    std::vector<int> degree(n, 0), offsets(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        if (a[i] >= 0) { ++degree[i]; ++offsets[a[i] + 1]; }
        if (b[i] >= 0) { ++degree[i]; ++offsets[b[i] + 1]; }
    }
    for (int i = 1; i <= n; ++i) offsets[i] += offsets[i - 1];
    auto cursor = offsets;
    std::vector<int> children(offsets[n]);
    for (int i = 0; i < n; ++i) {
        if (a[i] >= 0) children[cursor[a[i]]++] = i;
        if (b[i] >= 0) children[cursor[b[i]]++] = i;
    }
    int end = 0;
    for (int i = 0; i < n; ++i) if (!degree[i]) order[end++] = i;
    for (int at = 0; at < end; ++at) {
        const int node = order[at];
        for (int j = offsets[node]; j < offsets[node + 1]; ++j) {
            if (--degree[children[j]] == 0) order[end++] = children[j];
        }
    }
    return end;
}

extern "C" double fa_latency_eval(int n, int templates, int dimensions, int root,
    const int32_t *a, const int32_t *b, const int32_t *which,
    const double *matrix, const double *rates, double *gradient)
{
    std::vector<double> duration(templates, 0.0), finish(n, 0.0);
    std::vector<int> parent(gradient ? n : 0, -1);
    for (int t = 0; t < templates; ++t) {
        for (int j = 0; j < dimensions; ++j) duration[t] += matrix[t * dimensions + j] * rates[j];
    }
    for (int i = 0; i < n; ++i) {
        const double left = a[i] < 0 ? 0.0 : finish[a[i]];
        const double right = b[i] < 0 ? 0.0 : finish[b[i]];
        finish[i] = std::max(left, right) + duration[which[i]];
        if (gradient) parent[i] = left >= right ? a[i] : b[i];
    }
    if (gradient) {
        std::vector<double> counts(templates, 0.0);
        for (int node = root; node >= 0; node = parent[node]) ++counts[which[node]];
        for (int j = 0; j < dimensions; ++j) {
            gradient[j] = 0.0;
            for (int t = 0; t < templates; ++t) gradient[j] += counts[t] * matrix[t * dimensions + j];
        }
    }
    return finish[root];
}
