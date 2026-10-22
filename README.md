# binarypattern

Experimental binary sequence modelling with a hidden Markov model (HMM), based on [hmmm 0.2.0](https://crates.io/crates/hmmm).

## Build and tests

```sh
cargo build --locked
cargo test --locked
cargo clippy --locked
# or: just build
```

The `hmmm` crate uses `ndarray 0.13.1`. This project pins its direct `ndarray` dependency to the same minor version because Rust considers arrays from different `ndarray` versions distinct types.

## Example

```rust
use binarypattern::{HmmmModel, PredictFollowF32, TrainF32, TransformResult};

let values: Vec<f32> = (0..120).map(|i| (i % 2) as f32).collect();
let mut model = HmmmModel::<f32>::new(2, 2, 42);
let mut predictor = model.train(values.into_iter());
let mut out = [0.0; 8];
assert_eq!(predictor.predict(&mut out), TransformResult::Ok);
assert!(out.iter().all(|&x| x == 0.0 || x == 1.0));
```

`HmmmModel::new(observations_n, hidden_states_k, seed)` takes the number of distinct **observation symbols** first, then the number of **hidden states**. Observations are currently cast from `f32` to `usize`, so use nonnegative integer-coded symbols in the range `0..observations_n`.

**Important forecasting caveat:** `HmmmPredict::predict` starts the trained model's sampler to generate an observation sequence. It does **not** condition that sampler on the end of the training sequence and therefore is **not yet** a genuine next-symbol forecast conditioned on recent observations. The tests check valid observations, repeatability for a fixed seed and complete output. They do not establish predictive skill.

To evaluate forecasting performance, implement an HMM forward filter on the observed history and draw or select the next observation from the corresponding posterior predictive distribution. Then use time-ordered holdout data and compare against simple frequency and persistence baselines without leaking future values.
