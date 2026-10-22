use hmmm::HMM;
use rand_07::SeedableRng as _;
use rand_07::rngs::StdRng as TrainingRng;
use rand::rngs::StdRng;

use crate::*;

impl<T> HmmmModel<T> {
    pub fn new(observations_n: usize, hidden_states_k: usize, seed: u64) -> Self {
        HmmmModel {
            marker: Default::default(),
            observations_n,
            hidden_states_k,
            rng: TrainingRng::seed_from_u64(seed),
        }
    }
}

impl<T> HmmmPredict<T> {
    pub fn new(hmm: HMM, _rng: StdRng) -> Self {
        HmmmPredict {
            marker: Default::default(),
            hmm,
            history: Vec::new(),
        }
    }

    /// Forecast observation probabilities conditional on the supplied history.
    /// Row t contains P(Y_(T+t) = symbol | observed history), with t starting at 0.
    /// Future hidden-state distributions are propagated without observing future symbols.
    pub fn forecast_probabilities(&self, history: &[usize], steps: usize) -> Result<Vec<Vec<f64>>, String> {
        let states = self.hmm.n();
        let symbols = self.hmm.k();
        if states == 0 || symbols == 0 {
            return Err("empty HMM".into());
        }
        let mut posterior = self.hmm.pi.to_vec();
        for &symbol in history {
            if symbol >= symbols {
                return Err(format!("observation {symbol} is outside 0..{symbols}"));
            }
            // Update P(hidden state | observations so far) using Bayes' rule.
            let mut total = 0.0;
            for (state, probability) in posterior.iter_mut().enumerate() {
                *probability *= self.hmm.b[[state, symbol]];
                total += *probability;
            }
            if !total.is_finite() || total <= 0.0 {
                return Err("observation history has zero probability under the model".into());
            }
            for p in &mut posterior {
                *p /= total;
            }
            // Advance to the next hidden state before consuming the next observation.
            let mut next = vec![0.0; states];
            for (i, &probability) in posterior.iter().enumerate() {
                for (j, next_probability) in next.iter_mut().enumerate() {
                    *next_probability += probability * self.hmm.a[[i, j]];
                }
            }
            posterior = next;
        }

        let mut forecasts = Vec::with_capacity(steps);
        for _ in 0..steps {
            let mut observation_probs = vec![0.0; symbols];
            for (state, &probability) in posterior.iter().enumerate() {
                for (symbol, observation_probability) in observation_probs.iter_mut().enumerate() {
                    *observation_probability += probability * self.hmm.b[[state, symbol]];
                }
            }
            forecasts.push(observation_probs);
            let mut next = vec![0.0; states];
            for (i, &probability) in posterior.iter().enumerate() {
                for (j, next_probability) in next.iter_mut().enumerate() {
                    *next_probability += probability * self.hmm.a[[i, j]];
                }
            }
            posterior = next;
        }
        Ok(forecasts)
    }

    /// Replace the context used by the legacy no-argument prediction API.
    pub fn set_history(&mut self, history: &[usize]) {
        self.history = history.to_vec();
    }
}

impl<T> TrainF32<HmmmPredict<HMM>> for HmmmModel<T> {
    fn train<RI>(&mut self, ri: RI) -> HmmmPredict<HMM>
    where
        RI: PredictReadIterator<Item = f32>,
    {
        let data: Vec<usize> = ri.map(|x| x as usize).collect();
        let hmm = HMM::train(&ndarray_013::Array1::from(data.clone()), self.hidden_states_k, self.observations_n, &mut self.rng);
        let mut predictor = HmmmPredict::<HMM>::new(hmm, self.rng.clone());
        predictor.set_history(&data);
        predictor
    }
}

impl<T> PredictFollowF32<HMM> for HmmmPredict<T> {
    fn predict(&mut self, out: &mut [f32]) -> TransformResult {
        let predictions = match self.forecast_probabilities(&self.history, out.len()) {
            Ok(value) => value,
            Err(_) => return TransformResult::Error(1),
        };
        for (dest, probabilities) in out.iter_mut().zip(predictions) {
            // Deterministic most likely symbol; use forecast_probabilities for calibrated output.
            let best = probabilities.iter().enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(symbol, _)| symbol)
                .unwrap_or(0);
            *dest = best as f32;
        }
        TransformResult::Ok
    }
}
