#pragma once

#include <cstddef>
#include <vector>

namespace rflux::qflux {

/*
 * Buffer-first contract: every function has an `_into` overload that writes
 * exactly `n` values to caller-owned storage; `macd_into` and `bollinger_into`
 * write three independent buffers.  The C ABI and the Python bridge use the
 * buffer overloads so no intermediate result vector is allocated or copied.
 */

struct MacdResult {
    std::vector<double> macd;
    std::vector<double> signal;
    std::vector<double> histogram;
};

struct BollingerBands {
    std::vector<double> middle;
    std::vector<double> upper;
    std::vector<double> lower;
};

/* Simple or log returns. */
void returns_into(const double* prices, std::size_t n, bool log_returns,
                  double* out);
std::vector<double> returns(const double* prices, std::size_t n,
                            bool log_returns = false);

/* Alias for qflux::rolling_mean, exposed with the common indicator name. */
void sma_into(const double* values, std::size_t n, std::size_t window,
              double* out);
std::vector<double> sma(const double* values, std::size_t n, std::size_t window);

/*
 * EMA with alpha = 2/(span+1).  Seeded with the first finite value of every
 * contiguous finite run; non-finite values reset the recursion.
 */
void ema_into(const double* values, std::size_t n, std::size_t span,
              double* out);
std::vector<double> ema(const double* values, std::size_t n, std::size_t span);

/* Wilder RSI.  Flat windows map to 50; zero losses map to 100. */
void rsi_into(const double* prices, std::size_t n, std::size_t period,
              double* out);
std::vector<double> rsi(const double* prices, std::size_t n,
                        std::size_t period);

/* MACD with the same EMA convention as ema(). */
void macd_into(const double* prices, std::size_t n, std::size_t fast,
               std::size_t slow, std::size_t signal, double* macd_out,
               double* signal_out, double* histogram_out);
MacdResult macd(const double* prices, std::size_t n, std::size_t fast = 12,
                std::size_t slow = 26, std::size_t signal = 9);

/* SMA middle band plus population standard deviation bands. */
void bollinger_into(const double* prices, std::size_t n, std::size_t window,
                    double n_std, double* middle, double* upper,
                    double* lower);
BollingerBands bollinger(const double* prices, std::size_t n,
                         std::size_t window = 20, double n_std = 2.0);

/* Wilder average true range. */
void atr_into(const double* high, const double* low, const double* close,
              std::size_t n, std::size_t period, double* out);
std::vector<double> atr(const double* high, const double* low,
                        const double* close, std::size_t n,
                        std::size_t period = 14);

/* equity / running_peak - 1. */
void drawdown_into(const double* equity, std::size_t n, double* out);
std::vector<double> drawdown(const double* equity, std::size_t n);

}  // namespace rflux::qflux
