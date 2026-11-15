/*
 * Minimal C ABI example.  Build:
 *   cc -std=c11 -Iinclude examples/c_basic.c -Lbuild -lrflux \
 *      -Wl,-rpath,$PWD/build -o build/example_c_basic
 */
#include "rflux/rflux.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const size_t n = 80;
    double* prices = (double*)malloc(n * sizeof(double));
    double* sma = (double*)malloc(n * sizeof(double));
    double* rsi = (double*)malloc(n * sizeof(double));
    if (prices == NULL || sma == NULL || rsi == NULL) return 1;

    for (size_t i = 0; i < n; ++i) {
        prices[i] = 100.0 + 0.35 * (double)i +
                    3.0 * sin((double)i * 0.23);
    }

    if (rflux_sma(prices, n, 20, sma) != RFLUX_OK ||
        rflux_rsi(prices, n, 14, rsi) != RFLUX_OK) {
        fprintf(stderr, "rflux error: %s\n", rflux_last_error());
        free(prices);
        free(sma);
        free(rsi);
        return 1;
    }
    printf("rflux %s (ABI %u)\n", rflux_version_string(),
           rflux_abi_version());
    printf("last price %.3f | SMA20 %.3f | RSI14 %.2f\n", prices[n - 1],
           sma[n - 1], rsi[n - 1]);

    const double corr[9] = {
        1.0, 0.82, 0.18,
        0.82, 1.0, 0.22,
        0.18, 0.22, 1.0};
    int32_t labels[3];
    size_t components = 0;
    size_t largest = 0;
    if (rflux_graph_components(corr, 3, 0.5, labels, &components, &largest) !=
        RFLUX_OK) {
        fprintf(stderr, "rflux error: %s\n", rflux_last_error());
        return 1;
    }
    printf("threshold graph: %zu component(s), largest %zu\n", components,
           largest);

    free(prices);
    free(sma);
    free(rsi);
    return 0;
}
