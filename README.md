# binarypattern

Experimental Rust library for discrete-sequence prediction with Hidden Markov Models (HMMs).

The current implementation uses [hmmm](https://docs.rs/hmmm/0.2.0/hmmm/struct.HMM.html) to train a discrete HMM and forecast **future observations conditioned on an observed history**. It is an experimental building block, not a validated financial forecasting system.

## Why this change?

The original predictor called `HMM::sampler()`, which generates a new, unconditional sequence. Such samples do **not** answer "what comes next, given the tokens we just observed?"

The new `forecast_probabilities(history, steps)` performs normalized forward filtering over the history, then propagates hidden-state probabilities through the transition matrix. Each returned vector is a probability distribution over possible observation symbols at that future step.

## Usage

```rust
use binarypattern::*;
use hmmm::HMM;
use ndarray::array;
use rand::SeedableRng;

let hmm = HMM::new(
    array![[0.9, 0.1], [0.1, 0.9]], // hidden-state transitions
    array![[1.0, 0.0], [0.0, 1.0]], // emission probabilities
    array![0.5, 0.5],             // initial hidden-state probabilities
);
let rng = rand::rngs::StdRng::seed_from_u64(42);
let mut predictor = HmmmPredict::<HMM>::new(hmm, rng);

let probabilities = predictor.forecast_probabilities(&[0], 2).unwrap();
assert!((probabilities[0][0] - 0.9).abs() < 1e-10);

predictor.set_history(&[1]);
let mut next = [0.0; 2];
assert_eq!(predictor.predict(&mut next), TransformResult::Ok);
assert_eq!(next, [1.0, 1.0]);
```

## Training

`HmmmModel::new(observations_n, hidden_states_k, seed)` configures the number of distinct discrete symbols, hidden states, and RNG seed. The `TrainF32` implementation converts observations to symbol IDs and trains the HMM. The trained predictor retains the training observations as its initial forecasting context. Use `set_history` to change that context.

The legacy `predict(&mut out)` interface returns the **most likely symbol at each forecast horizon**, not random samples. Prefer `forecast_probabilities` if uncertainty matters. These multi-step forecasts marginalize over unobserved future states and symbols; they do not feed hard predictions back into the model.

**Important:** Inputs must be discrete, nonnegative integer symbol IDs in `0..observations_n`. The legacy f32 training interface currently casts values to integers; validate or encode them before passing them in. `forecast_probabilities` rejects out-of-range symbols and zero-likelihood histories.

## Tests

```sh
cargo test --locked
cargo clippy --locked
```

Tests cover context-sensitive probabilities, multi-step propagation, normalization, invalid observations, training, and edge cases such as empty forecasts and invalid prediction history.\n\n`hmmm 0.2.0` depends on `ndarray 0.13.1`; this project uses the same `ndarray` version so its array types match the training API.

## Limitations

- Discrete observations only; no continuous emissions.
- HMM fitting may converge to local optima.
- A fitted HMM does not imply useful stock-price prediction. Evaluate on strictly time-ordered out-of-sample data, with no future information in feature extraction or model fitting.
