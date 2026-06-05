# ternary-prophet: Prediction and forecasting with uncertainty for ternary state systems

Trend detection, seasonality analysis, ensemble forecasting, accuracy tracking, and model evolution for {-1, 0, +1} state sequences.

## Why This Exists

Ternary systems change over time, and if you can't predict where they're heading, you're always reacting instead of planning. This crate provides forecasting tools that understand ternary data specifically — not continuous values squeezed into three buckets, but native {-1, 0, +1} sequences with their own statistical properties. It detects trends, finds seasonal cycles, combines multiple forecasts, tracks accuracy, and even evolves better models over time.

## Core Concepts

- **Ternary**: A value in {-1, 0, +1}. Negative, Zero, or Positive.
- **Prediction**: A forecast with a predicted ternary value, confidence level, and confidence interval (lower/upper bounds).
- **Trend**: Direction of data movement — Rising (toward +1), Falling (toward -1), Stable (near 0), or Flat (no clear trend).
- **TrendDetector**: Fits linear regression over a sliding window to identify trend direction and strength.
- **SeasonalPattern**: A detected repeating cycle with period, pattern, and strength (from autocorrelation).
- **Seasonality**: Detects cyclic patterns using autocorrelation at candidate periods.
- **Prophet**: The main forecasting engine combining trend and seasonality with confidence decay over horizon.
- **ProphetEnsemble**: Combines multiple prophets with weighted voting for robust predictions.
- **AccuracyTracker**: Records predictions vs actuals, computes accuracy, recent accuracy, and confusion matrices.
- **ProphetEvolver**: Evolves prophet parameters (confidence decay) using selection and mutation.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-prophet = "0.1"
```

```rust
use ternary_prophet::*;

// Create a prophet with confidence decay
let mut prophet = Prophet::new(0.9);

// Feed historical data
prophet.observe(&[
    Ternary::Negative, Ternary::Negative, Ternary::Zero,
    Ternary::Zero, Ternary::Positive,
]);

// Check what was detected
println!("Trend: {:?}", prophet.trend());

// Predict 3 steps ahead
let pred = prophet.predict(3);
println!("Predicted: {:?}, Confidence: {:.2}", pred.predicted, pred.confidence);
println!("Interval: {:?} to {:?}", pred.lower_bound, pred.upper_bound);

// Track accuracy
let mut tracker = AccuracyTracker::new();
tracker.record(pred.predicted, Ternary::Positive);
println!("Accuracy: {:.1}%", tracker.accuracy() * 100.0);
```

## API Overview

| Type | Description |
|------|-------------|
| `Prophet` | Main forecasting engine with trend and seasonality |
| `Prediction` | A forecast with confidence interval and horizon |
| `TrendDetector` | Identifies trends via linear regression |
| `Seasonality` | Detects cyclic patterns via autocorrelation |
| `SeasonalPattern` | A detected cycle with period, pattern, strength |
| `ProphetEnsemble` | Weighted combination of multiple prophets |
| `AccuracyTracker` | Records predictions vs actuals, computes metrics |
| `ProphetEvolver` | Evolves prophet parameters for better forecasts |

## How It Works

**Trend detection** fits ordinary least squares linear regression to the ternary values (treated as -1, 0, +1) over a sliding window. The slope determines the trend: > 0.2 = Rising, < -0.2 = Falling, near-zero with low variance = Stable, otherwise Flat. Strength is the absolute slope clamped to [0, 1].

**Seasonality detection** uses autocorrelation: for each candidate period p, it computes the correlation between the sequence and its p-shifted version. The period with the highest correlation above 0.3 is selected. The pattern is simply the first period-length segment of the data.

**Prophet** combines both: if a strong seasonal pattern is detected (strength > 0.5), it uses the seasonal forecast; otherwise it extrapolates from the linear trend. Confidence decays exponentially with the prediction horizon using the configurable decay parameter.

**ProphetEnsemble** runs multiple prophets and takes a weighted vote on the predicted ternary value. The combined confidence is the weighted average of individual confidences.

**ProphetEvolver** maintains a population of confidence-decay values, evaluates each on a train/test split of historical data, keeps the top half, and mutates copies for the next generation.

## Known Limitations

- Linear regression for trend detection assumes monotonic change; it cannot capture inflection points.
- Autocorrelation-based seasonality requires at least 2 full cycles in the data to detect a pattern.
- Confidence intervals widen quickly with horizon — beyond ~5 steps, predictions are essentially unbounded.
- The evolver's mutation strategy is simple (jitter around parents); it may not escape local optima.
- No support for exogenous variables or multi-variate forecasting.
- All data must fit in memory; no streaming or windowed training.
- The ensemble uses simple weighted voting; more sophisticated combination methods (e.g., Bayesian model averaging) are not available.

## Use Cases

- **Capacity planning**: Predict whether system load will trend positive, negative, or stay flat over the next N periods.
- **Sentiment forecasting**: Predict future ternary sentiment (negative/neutral/positive) from historical patterns.
- **Trading signals**: Forecast buy/hold/sell signals with confidence intervals for risk assessment.
- **Anomaly anticipation**: Detect seasonal patterns so you know when to expect spikes in negative states.
- **Model selection**: Use the evolver to find optimal forecasting parameters for a specific ternary dataset.

## Ecosystem Context

Part of the SuperInstance ternary ecosystem. Consumes historical data from `ternary-chronicle`. Predictions can be validated with `AccuracyTracker` and fed back to the evolver for continuous improvement. Works alongside `ternary-compass` for real-time heading estimation (prophet for long-range, compass for short-range).

## License

MIT
