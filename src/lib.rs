#![forbid(unsafe_code)]

//! Prediction and forecasting with uncertainty for ternary state systems.
//!
//! Provides prophet-based forecasting for {-1, 0, +1} state sequences:
//! trend detection, seasonality analysis, ensemble forecasting, accuracy
//! tracking, and model evolution.

use std::collections::HashMap;

/// A ternary value: Negative (-1), Zero (0), or Positive (+1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ternary {
    Negative,
    Zero,
    Positive,
}

impl Ternary {
    pub fn value(self) -> i8 {
        match self {
            Ternary::Negative => -1,
            Ternary::Zero => 0,
            Ternary::Positive => 1,
        }
    }

    pub fn from_value(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Ternary::Negative),
            0 => Some(Ternary::Zero),
            1 => Some(Ternary::Positive),
            _ => None,
        }
    }
}

// ─── Prediction ──────────────────────────────────────────────────────

/// A forecast with confidence interval.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Predicted value.
    pub predicted: Ternary,
    /// Confidence level (0.0 to 1.0).
    pub confidence: f64,
    /// Lower bound of confidence interval (most pessimistic).
    pub lower_bound: Ternary,
    /// Upper bound of confidence interval (most optimistic).
    pub upper_bound: Ternary,
    /// Number of steps ahead this prediction is for.
    pub horizon: usize,
}

impl Prediction {
    /// Create a prediction with the same bounds as the predicted value (narrow interval).
    pub fn narrow(predicted: Ternary, confidence: f64, horizon: usize) -> Self {
        Prediction {
            predicted,
            confidence,
            lower_bound: predicted,
            upper_bound: predicted,
            horizon,
        }
    }

    /// Create a prediction with a wide interval.
    pub fn wide(predicted: Ternary, confidence: f64, horizon: usize) -> Self {
        Prediction {
            predicted,
            confidence,
            lower_bound: Ternary::Negative,
            upper_bound: Ternary::Positive,
            horizon,
        }
    }

    /// Check if a value falls within the confidence interval.
    pub fn contains(&self, value: Ternary) -> bool {
        value.value() >= self.lower_bound.value() && value.value() <= self.upper_bound.value()
    }
}

// ─── TrendDetector ───────────────────────────────────────────────────

/// Identifies trends in ternary data.
#[derive(Debug, Clone)]
pub enum Trend {
    /// Values trending toward +1.
    Rising,
    /// Values trending toward -1.
    Falling,
    /// Values stable around 0.
    Stable,
    /// No clear trend.
    Flat,
}

impl PartialEq for Trend {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

/// Detects trends in ternary state sequences.
pub struct TrendDetector {
    /// Window size for trend detection.
    window_size: usize,
}

impl TrendDetector {
    pub fn new(window_size: usize) -> Self {
        TrendDetector {
            window_size: window_size.max(3),
        }
    }

    /// Detect the current trend in a data window.
    pub fn detect(&self, data: &[Ternary]) -> Trend {
        if data.len() < 3 {
            return Trend::Flat;
        }

        let window: Vec<Ternary> = if data.len() > self.window_size {
            data[data.len() - self.window_size..].to_vec()
        } else {
            data.to_vec()
        };

        // Linear regression on window
        let n = window.len() as f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xy = 0.0f64;
        let mut sum_xx = 0.0f64;

        for (i, t) in window.iter().enumerate() {
            let x = i as f64;
            let y = t.value() as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return Trend::Flat;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;

        if slope > 0.2 {
            Trend::Rising
        } else if slope < -0.2 {
            Trend::Falling
        } else {
            let variance = compute_variance(&window);
            if variance < 0.1 {
                Trend::Stable
            } else {
                Trend::Flat
            }
        }
    }

    /// Detect trend strength (0.0 to 1.0).
    pub fn strength(&self, data: &[Ternary]) -> f64 {
        if data.len() < 3 {
            return 0.0;
        }

        let window: Vec<Ternary> = if data.len() > self.window_size {
            data[data.len() - self.window_size..].to_vec()
        } else {
            data.to_vec()
        };

        let n = window.len() as f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xy = 0.0f64;
        let mut sum_xx = 0.0f64;

        for (i, t) in window.iter().enumerate() {
            let x = i as f64;
            let y = t.value() as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return 0.0;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        slope.abs().min(1.0)
    }
}

fn compute_variance(data: &[Ternary]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len() as f64;
    let mean = data.iter().map(|t| t.value() as f64).sum::<f64>() / n;
    data.iter().map(|t| (t.value() as f64 - mean).powi(2)).sum::<f64>() / n
}

// ─── Seasonality ─────────────────────────────────────────────────────

/// Detects cyclic patterns in ternary data.
#[derive(Debug, Clone)]
pub struct SeasonalPattern {
    /// Detected period (number of steps per cycle).
    pub period: usize,
    /// The repeating pattern.
    pub pattern: Vec<Ternary>,
    /// Strength of the seasonal signal (0.0 to 1.0).
    pub strength: f64,
}

/// Analyzes ternary data for seasonal/cyclic patterns.
pub struct Seasonality;

impl Seasonality {
    /// Detect the dominant period using autocorrelation.
    pub fn detect_period(data: &[Ternary], min_period: usize, max_period: usize) -> Option<SeasonalPattern> {
        if data.len() < min_period * 2 {
            return None;
        }

        let values: Vec<f64> = data.iter().map(|t| t.value() as f64).collect();
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = values.iter().map(|v| v - mean).collect();
        let variance: f64 = centered.iter().map(|v| v * v).sum();

        if variance < f64::EPSILON {
            return None;
        }

        let mut best_period = 0;
        let mut best_corr = 0.0f64;

        for period in min_period..=max_period.min(n / 2) {
            let corr: f64 = centered.iter().enumerate()
                .take(n - period)
                .map(|(i, &v)| v * centered[i + period])
                .sum::<f64>()
                / variance;
            if corr > best_corr {
                best_corr = corr;
                best_period = period;
            }
        }

        if best_period == 0 || best_corr < 0.3 {
            return None;
        }

        // Extract the pattern
        let pattern: Vec<Ternary> = data[..best_period].to_vec();
        Some(SeasonalPattern {
            period: best_period,
            pattern,
            strength: best_corr,
        })
    }

    /// Forecast n steps ahead using a detected seasonal pattern.
    pub fn forecast(pattern: &SeasonalPattern, steps_ahead: usize, data_len: usize) -> Ternary {
        let idx = (data_len + steps_ahead - 1) % pattern.period;
        pattern.pattern[idx]
    }
}

// ─── Prophet ─────────────────────────────────────────────────────────

/// The main forecasting engine for ternary state sequences.
#[derive(Debug, Clone)]
pub struct Prophet {
    /// Historical data.
    data: Vec<Ternary>,
    /// Detected trend.
    trend: Option<Trend>,
    /// Detected seasonality.
    seasonality: Option<SeasonalPattern>,
    /// Prediction confidence decay per step.
    confidence_decay: f64,
}

impl Prophet {
    pub fn new(confidence_decay: f64) -> Self {
        Prophet {
            data: Vec::new(),
            trend: None,
            seasonality: None,
            confidence_decay: confidence_decay.clamp(0.0, 1.0),
        }
    }

    /// Add observations.
    pub fn observe(&mut self, values: &[Ternary]) {
        self.data.extend_from_slice(values);
        self.reanalyze();
    }

    /// Re-analyze the data for trends and seasonality.
    pub fn reanalyze(&mut self) {
        if self.data.len() >= 3 {
            let detector = TrendDetector::new(10);
            self.trend = Some(detector.detect(&self.data));
        }
        if self.data.len() >= 6 {
            self.seasonality = Seasonality::detect_period(&self.data, 2, self.data.len() / 2);
        }
    }

    /// Generate a prediction for n steps ahead.
    pub fn predict(&self, horizon: usize) -> Prediction {
        if self.data.is_empty() {
            return Prediction::narrow(Ternary::Zero, 0.5, horizon);
        }

        // Trend-based prediction
        let trend_pred = self.predict_from_trend(horizon);
        // Seasonality-based prediction
        let seasonal_pred = self.seasonality.as_ref()
            .map(|s| Seasonality::forecast(s, horizon, self.data.len()));

        // Combine: prefer seasonal if strong, else use trend
        let (predicted, confidence) = if let Some(sp) = seasonal_pred {
            if let Some(ref s) = self.seasonality {
                if s.strength > 0.5 {
                    (sp, s.strength * self.confidence_decay.powi(horizon as i32))
                } else {
                    (trend_pred.0, trend_pred.1)
                }
            } else {
                (trend_pred.0, trend_pred.1)
            }
        } else {
            trend_pred
        };

        // Compute confidence interval
        let (lower, upper) = if confidence > 0.7 {
            (predicted, predicted)
        } else if confidence > 0.4 {
            (Ternary::from_value(predicted.value().min(0) - 1).unwrap_or(Ternary::Negative),
             Ternary::from_value(predicted.value().max(0) + 1).unwrap_or(Ternary::Positive))
        } else {
            (Ternary::Negative, Ternary::Positive)
        };

        Prediction {
            predicted,
            confidence: confidence.max(0.0).min(1.0),
            lower_bound: lower,
            upper_bound: upper,
            horizon,
        }
    }

    fn predict_from_trend(&self, horizon: usize) -> (Ternary, f64) {
        if self.data.len() < 2 {
            return (*self.data.last().unwrap_or(&Ternary::Zero), 0.5);
        }

        // Simple linear extrapolation
        let n = self.data.len() as f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xy = 0.0f64;
        let mut sum_xx = 0.0f64;

        for (i, t) in self.data.iter().enumerate() {
            let x = i as f64;
            let y = t.value() as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return (*self.data.last().unwrap(), 0.5);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;
        let predicted_val = intercept + slope * (self.data.len() + horizon - 1) as f64;

        let predicted = if predicted_val > 0.33 {
            Ternary::Positive
        } else if predicted_val < -0.33 {
            Ternary::Negative
        } else {
            Ternary::Zero
        };

        // Confidence decays with distance
        let confidence = 0.8 * self.confidence_decay.powi(horizon as i32);

        (predicted, confidence)
    }

    /// Get the current data length.
    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    /// Get the detected trend.
    pub fn trend(&self) -> Option<&Trend> {
        self.trend.as_ref()
    }

    /// Get the detected seasonality.
    pub fn seasonality(&self) -> Option<&SeasonalPattern> {
        self.seasonality.as_ref()
    }
}

// ─── ProphetEnsemble ─────────────────────────────────────────────────

/// Combines multiple prophets for better forecasts.
#[derive(Debug, Clone)]
pub struct ProphetEnsemble {
    prophets: Vec<Prophet>,
    weights: Vec<f64>,
}

impl ProphetEnsemble {
    pub fn new() -> Self {
        ProphetEnsemble {
            prophets: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Add a prophet with a weight.
    pub fn add_prophet(&mut self, prophet: Prophet, weight: f64) {
        self.prophets.push(prophet);
        self.weights.push(weight);
    }

    /// Generate an ensemble prediction.
    pub fn predict(&self, horizon: usize) -> Prediction {
        if self.prophets.is_empty() {
            return Prediction::narrow(Ternary::Zero, 0.0, horizon);
        }

        let total_weight: f64 = self.weights.iter().sum();
        if total_weight < f64::EPSILON {
            return Prediction::narrow(Ternary::Zero, 0.0, horizon);
        }

        // Weighted vote for predicted value
        let mut scores: HashMap<i8, f64> = HashMap::new();
        let mut weighted_confidence = 0.0;

        for (prophet, &weight) in self.prophets.iter().zip(self.weights.iter()) {
            let pred = prophet.predict(horizon);
            let w = weight / total_weight;
            *scores.entry(pred.predicted.value()).or_default() += w;
            weighted_confidence += pred.confidence * w;
        }

        let best_value = scores.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&v, _)| v)
            .unwrap_or(0);

        let predicted = Ternary::from_value(best_value).unwrap_or(Ternary::Zero);

        Prediction {
            predicted,
            confidence: weighted_confidence,
            lower_bound: Ternary::Negative,
            upper_bound: Ternary::Positive,
            horizon,
        }
    }

    /// Number of prophets in the ensemble.
    pub fn len(&self) -> usize {
        self.prophets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prophets.is_empty()
    }
}

impl Default for ProphetEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

// ─── AccuracyTracker ─────────────────────────────────────────────────

/// Tracks prediction accuracy over time.
#[derive(Debug, Clone)]
pub struct AccuracyTracker {
    /// (predicted, actual) pairs.
    history: Vec<(Ternary, Ternary)>,
    /// Running accuracy.
    correct: usize,
    total: usize,
}

impl AccuracyTracker {
    pub fn new() -> Self {
        AccuracyTracker {
            history: Vec::new(),
            correct: 0,
            total: 0,
        }
    }

    /// Record a prediction vs actual outcome.
    pub fn record(&mut self, predicted: Ternary, actual: Ternary) {
        if predicted == actual {
            self.correct += 1;
        }
        self.total += 1;
        self.history.push((predicted, actual));
    }

    /// Current accuracy (0.0 to 1.0).
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.correct as f64 / self.total as f64
    }

    /// Accuracy within the last n predictions.
    pub fn recent_accuracy(&self, n: usize) -> f64 {
        let start = self.total.saturating_sub(n);
        let recent = &self.history[start..];
        if recent.is_empty() {
            return 0.0;
        }
        let correct = recent.iter().filter(|(p, a)| p == a).count();
        correct as f64 / recent.len() as f64
    }

    /// Confusion matrix: predicted vs actual counts.
    pub fn confusion_matrix(&self) -> [[usize; 3]; 3] {
        let mut matrix = [[0usize; 3]; 3];
        for (pred, actual) in &self.history {
            let pi = (pred.value() + 1) as usize;
            let ai = (actual.value() + 1) as usize;
            matrix[pi][ai] += 1;
        }
        matrix
    }

    /// Total predictions recorded.
    pub fn total(&self) -> usize {
        self.total
    }
}

impl Default for AccuracyTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ProphetEvolver ──────────────────────────────────────────────────

/// Evolves better forecasting models by tuning parameters.
#[derive(Debug, Clone)]
pub struct ProphetEvolver {
    /// Population of (confidence_decay, fitness) pairs.
    population: Vec<(f64, f64)>,
    /// Best decay found so far.
    best_decay: f64,
    /// Best fitness.
    best_fitness: f64,
    /// Number of generations evolved.
    generations: usize,
}

impl ProphetEvolver {
    pub fn new(population_size: usize) -> Self {
        let mut population = Vec::new();
        for _ in 0..population_size {
            let decay = 0.5 + 0.5 * (population.len() as f64 / population_size as f64);
            population.push((decay, 0.0));
        }
        ProphetEvolver {
            population,
            best_decay: 0.9,
            best_fitness: 0.0,
            generations: 0,
        }
    }

    /// Evaluate a prophet's fitness on training data.
    pub fn evaluate(&mut self, data: &[Ternary], test_split: f64) -> f64 {
        if data.len() < 4 {
            return 0.0;
        }

        let split_idx = ((data.len() as f64) * (1.0 - test_split)) as usize;
        let train = &data[..split_idx];
        let test = &data[split_idx..];

        let mut best_fitness = 0.0f64;
        let mut best_decay = self.best_decay;

        for (decay, fitness) in &mut self.population {
            let mut prophet = Prophet::new(*decay);
            prophet.observe(train);

            let mut correct = 0usize;
            for (i, &actual) in test.iter().enumerate() {
                let pred = prophet.predict(1);
                if pred.predicted == actual {
                    correct += 1;
                }
                prophet.observe(&[actual]);
            }

            let accuracy = if test.is_empty() { 0.0 } else { correct as f64 / test.len() as f64 };
            *fitness = accuracy;

            if accuracy > best_fitness {
                best_fitness = accuracy;
                best_decay = *decay;
            }
        }

        self.best_fitness = best_fitness;
        self.best_decay = best_decay;
        best_fitness
    }

    /// Evolve one generation: mutate and select.
    pub fn evolve(&mut self) {
        // Sort by fitness descending
        self.population.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep top half, mutate copies
        let keep = self.population.len() / 2;
        let top: Vec<(f64, f64)> = self.population[..keep].to_vec();

        for i in keep..self.population.len() {
            let parent_idx = i % top.len();
            let parent_decay = top[parent_idx].0;
            // Gaussian-ish mutation
            let mutation = (self.generations as f64 * 0.01 + 0.05) * ((i as f64 * 7.3 + 1.5) % 1.0 - 0.5);
            let new_decay = (parent_decay + mutation).clamp(0.1, 0.99);
            self.population[i] = (new_decay, 0.0);
        }

        self.generations += 1;
    }

    /// Get the best decay parameter found.
    pub fn best_decay(&self) -> f64 {
        self.best_decay
    }

    /// Get the best fitness achieved.
    pub fn best_fitness(&self) -> f64 {
        self.best_fitness
    }

    /// Build a prophet with the best parameters.
    pub fn best_prophet(&self) -> Prophet {
        Prophet::new(self.best_decay)
    }

    /// Number of generations evolved.
    pub fn generations(&self) -> usize {
        self.generations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_narrow() {
        let p = Prediction::narrow(Ternary::Positive, 0.9, 1);
        assert_eq!(p.predicted, Ternary::Positive);
        assert_eq!(p.lower_bound, Ternary::Positive);
        assert!(p.contains(Ternary::Positive));
        assert!(!p.contains(Ternary::Negative));
    }

    #[test]
    fn test_prediction_wide() {
        let p = Prediction::wide(Ternary::Zero, 0.5, 3);
        assert!(p.contains(Ternary::Negative));
        assert!(p.contains(Ternary::Zero));
        assert!(p.contains(Ternary::Positive));
    }

    #[test]
    fn test_trend_detector_rising() {
        let detector = TrendDetector::new(10);
        let data = vec![Ternary::Negative, Ternary::Negative, Ternary::Zero, Ternary::Zero, Ternary::Positive];
        assert_eq!(detector.detect(&data), Trend::Rising);
    }

    #[test]
    fn test_trend_detector_falling() {
        let detector = TrendDetector::new(10);
        let data = vec![Ternary::Positive, Ternary::Positive, Ternary::Zero, Ternary::Zero, Ternary::Negative];
        assert_eq!(detector.detect(&data), Trend::Falling);
    }

    #[test]
    fn test_trend_detector_stable() {
        let detector = TrendDetector::new(10);
        let data = vec![Ternary::Zero, Ternary::Zero, Ternary::Zero, Ternary::Zero, Ternary::Zero];
        assert_eq!(detector.detect(&data), Trend::Stable);
    }

    #[test]
    fn test_trend_detector_too_short() {
        let detector = TrendDetector::new(10);
        assert_eq!(detector.detect(&[Ternary::Positive]), Trend::Flat);
    }

    #[test]
    fn test_trend_strength() {
        let detector = TrendDetector::new(10);
        let rising = vec![Ternary::Negative, Ternary::Negative, Ternary::Zero, Ternary::Zero, Ternary::Positive];
        let flat = vec![Ternary::Zero, Ternary::Zero, Ternary::Zero, Ternary::Zero, Ternary::Zero];
        assert!(detector.strength(&rising) > detector.strength(&flat));
    }

    #[test]
    fn test_seasonality_detect() {
        let data = vec![
            Ternary::Positive, Ternary::Zero, Ternary::Negative,
            Ternary::Positive, Ternary::Zero, Ternary::Negative,
            Ternary::Positive, Ternary::Zero, Ternary::Negative,
        ];
        let pattern = Seasonality::detect_period(&data, 2, 4);
        assert!(pattern.is_some());
        let pattern = pattern.unwrap();
        assert_eq!(pattern.period, 3);
        assert!(pattern.strength > 0.3);
    }

    #[test]
    fn test_seasonality_no_pattern() {
        let data = vec![Ternary::Positive, Ternary::Negative, Ternary::Zero, Ternary::Positive];
        let pattern = Seasonality::detect_period(&data, 2, 4);
        // Too short for reliable detection
        assert!(pattern.is_none() || pattern.unwrap().strength < 0.5);
    }

    #[test]
    fn test_seasonality_forecast() {
        let pattern = SeasonalPattern {
            period: 3,
            pattern: vec![Ternary::Positive, Ternary::Zero, Ternary::Negative],
            strength: 0.8,
        };
        assert_eq!(Seasonality::forecast(&pattern, 1, 3), Ternary::Positive); // idx=(3+1-1)%3=0
        assert_eq!(Seasonality::forecast(&pattern, 2, 3), Ternary::Zero); // idx=(3+2-1)%3=1
        assert_eq!(Seasonality::forecast(&pattern, 3, 3), Ternary::Negative); // idx=(3+3-1)%3=2
    }

    #[test]
    fn test_prophet_observe_and_predict() {
        let mut prophet = Prophet::new(0.9);
        prophet.observe(&[Ternary::Negative, Ternary::Zero, Ternary::Positive]);
        let pred = prophet.predict(1);
        assert_eq!(pred.horizon, 1);
        assert!(pred.confidence > 0.0);
    }

    #[test]
    fn test_prophet_empty_predict() {
        let prophet = Prophet::new(0.9);
        let pred = prophet.predict(1);
        assert_eq!(pred.predicted, Ternary::Zero);
    }

    #[test]
    fn test_prophet_trend_detected() {
        let mut prophet = Prophet::new(0.9);
        prophet.observe(&[Ternary::Negative, Ternary::Negative, Ternary::Zero, Ternary::Zero, Ternary::Positive]);
        assert!(prophet.trend().is_some());
    }

    #[test]
    fn test_prophet_seasonality_detected() {
        let mut prophet = Prophet::new(0.9);
        let data = vec![
            Ternary::Positive, Ternary::Zero, Ternary::Negative,
            Ternary::Positive, Ternary::Zero, Ternary::Negative,
            Ternary::Positive, Ternary::Zero,
        ];
        prophet.observe(&data);
        assert!(prophet.seasonality().is_some());
    }

    #[test]
    fn test_ensemble_predict() {
        let mut ensemble = ProphetEnsemble::new();
        let mut p1 = Prophet::new(0.9);
        p1.observe(&[Ternary::Negative, Ternary::Zero, Ternary::Positive]);
        let mut p2 = Prophet::new(0.7);
        p2.observe(&[Ternary::Negative, Ternary::Zero, Ternary::Positive]);

        ensemble.add_prophet(p1, 1.0);
        ensemble.add_prophet(p2, 0.5);

        let pred = ensemble.predict(1);
        assert_eq!(pred.horizon, 1);
        assert!(!ensemble.is_empty());
    }

    #[test]
    fn test_ensemble_empty() {
        let ensemble = ProphetEnsemble::new();
        assert!(ensemble.is_empty());
        let pred = ensemble.predict(1);
        assert_eq!(pred.confidence, 0.0);
    }

    #[test]
    fn test_accuracy_tracker() {
        let mut tracker = AccuracyTracker::new();
        tracker.record(Ternary::Positive, Ternary::Positive);
        tracker.record(Ternary::Negative, Ternary::Positive);
        tracker.record(Ternary::Zero, Ternary::Zero);
        assert!((tracker.accuracy() - 0.6667).abs() < 0.01);
    }

    #[test]
    fn test_accuracy_tracker_perfect() {
        let mut tracker = AccuracyTracker::new();
        tracker.record(Ternary::Positive, Ternary::Positive);
        tracker.record(Ternary::Positive, Ternary::Positive);
        assert!((tracker.accuracy() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_accuracy_tracker_recent() {
        let mut tracker = AccuracyTracker::new();
        tracker.record(Ternary::Positive, Ternary::Negative); // miss
        tracker.record(Ternary::Positive, Ternary::Positive); // hit
        assert!((tracker.recent_accuracy(1) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_accuracy_confusion_matrix() {
        let mut tracker = AccuracyTracker::new();
        tracker.record(Ternary::Positive, Ternary::Positive);
        tracker.record(Ternary::Negative, Ternary::Negative);
        let matrix = tracker.confusion_matrix();
        assert_eq!(matrix[2][2], 1); // Positive predicted Positive
        assert_eq!(matrix[0][0], 1); // Negative predicted Negative
    }

    #[test]
    fn test_evolver() {
        let mut evolver = ProphetEvolver::new(5);
        let data = vec![Ternary::Negative, Ternary::Zero, Ternary::Positive,
                        Ternary::Negative, Ternary::Zero, Ternary::Positive,
                        Ternary::Negative, Ternary::Zero, Ternary::Positive];
        let fitness = evolver.evaluate(&data, 0.3);
        assert!(fitness >= 0.0);
        assert!(evolver.best_decay() > 0.0);
    }

    #[test]
    fn test_evolver_evolve() {
        let mut evolver = ProphetEvolver::new(6);
        let data = vec![Ternary::Negative, Ternary::Zero, Ternary::Positive,
                        Ternary::Negative, Ternary::Zero, Ternary::Positive,
                        Ternary::Negative, Ternary::Zero, Ternary::Positive];
        evolver.evaluate(&data, 0.3);
        evolver.evolve();
        assert_eq!(evolver.generations(), 1);
        let prophet = evolver.best_prophet();
        assert_eq!(prophet.data_len(), 0); // new prophet, no data yet
    }

    #[test]
    fn test_ensemble_len() {
        let mut ensemble = ProphetEnsemble::new();
        assert_eq!(ensemble.len(), 0);
        ensemble.add_prophet(Prophet::new(0.9), 1.0);
        assert_eq!(ensemble.len(), 1);
    }
}
