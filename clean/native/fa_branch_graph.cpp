#include <cstdint>
#include <vector>
#include <algorithm>

extern "C" int fa_compact_graph(int n, int root, const int32_t *a, const int32_t *b,
    const int32_t *labels, const uint8_t *zero, int32_t *out_a, int32_t *out_b,
    int32_t *out_labels, int32_t *out_root)
{
    std::vector<int32_t> mapped(n, -1);
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (a[i] >= i || b[i] >= i) return -1;
        int left = a[i] < 0 ? -1 : mapped[a[i]];
        int right = b[i] < 0 ? -1 : mapped[b[i]];
        if (left == right) right = -1;
        if (left >= 0 && right >= 0) {
            if (out_a[right] == left || out_b[right] == left) left = -1;
            else if (out_a[left] == right || out_b[left] == right) right = -1;
        }
        if (left < 0) std::swap(left, right);
        if (zero[labels[i]] && right < 0) {
            mapped[i] = left;
            continue;
        }
        mapped[i] = count;
        out_a[count] = left;
        out_b[count] = right;
        out_labels[count] = labels[i];
        ++count;
    }
    *out_root = mapped[root];
    return count;
}
