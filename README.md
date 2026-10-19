# Intraday S&P 500 Trading Research

## Overview

This project investigates whether short-term price and volume patterns in S&P 500-related intraday market data can produce a measurable trading edge.

The goal was to systematically test trading hypotheses using historical data, evaluate the robustness of observed patterns, and determine whether any apparent edge could remain profitable after realistic transaction costs.

## Research Approach

The project progressed through multiple stages of experimentation, including:

- Testing intraday price and abnormal-volume patterns for potential continuation and reversal effects.
- Measuring returns over different holding periods and evaluating alternative entry assumptions.
- Examining whether results remained consistent when restricting the analysis to regular trading hours.
- Evaluating transaction costs and the sensitivity of results to execution assumptions.
- Comparing gross returns with estimated net returns to distinguish statistical patterns from potentially viable trading opportunities.

## Key Findings

The research identified a small short-term downside-reversal effect in the tested data. However, the effect was too small to support a viable standalone strategy under the transaction-cost assumptions examined.

Gross returns increased from approximately 0.013% at five minutes to 0.056% at thirty minutes. After accounting for five basis points of transaction costs, the thirty-minute result fell to approximately 0.006%; at ten basis points, the result became negative. Restricting the analysis to regular trading hours reduced the thirty-minute gross result to approximately 0.0002%.

The results suggested that transaction costs, rather than entry timing alone, were the primary obstacle to making the observed pattern tradable.

## Conclusion

The central takeaway was that detecting a repeatable market pattern does not necessarily translate into a profitable trading strategy. The observed effects were too small to justify continuing the strategy as a standalone approach under the tested assumptions.

This project emphasized systematic experimentation, realistic evaluation of trading costs, and using empirical results to decide when a hypothesis should be rejected rather than continuing to optimize an unpromising strategy.

## Reproducibility

The repository contains the staged research workflow, analysis scripts, and results used to evaluate the hypothesis. The final research stage includes `tobin/REPORT_STAGE4.md`, `results_stage4.csv`, `results_stage4_rth.csv`, and `stage4.py`.

The conclusions apply to the tested data, methodology, and cost assumptions; they do not establish that all intraday trading strategies are unprofitable.
