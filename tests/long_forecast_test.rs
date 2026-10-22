use binarypattern::{HmmmPredict, PredictFollowF32, TransformResult};
use hmmm::HMM;
use ndarray_013::array;
use rand::{rngs::StdRng, SeedableRng};

fn persistent_binary_model() -> HmmmPredict<HMM> {
    // A symmetric two-state Markov chain with perfectly informative emissions.
    // Its t-step persistence has the closed form 0.5 + 0.5 * 0.8^t.
    let hmm = HMM::new(
        array![[0.9, 0.1], [0.1, 0.9]],
        array![[1.0, 0.0], [0.0, 1.0]],
        array![0.5, 0.5],
    );
    HmmmPredict::new(hmm, StdRng::seed_from_u64(1234))
}

#[test]
fn forecasts_256_steps_match_analytic_markov_chain() {
    let predictor = persistent_binary_model();
    let forecasts = predictor.forecast_probabilities(&[0], 256).unwrap();
    assert_eq!(forecasts.len(), 256);

    for (horizon, probs) in forecasts.iter().enumerate() {
        let expected_zero = 0.5 + 0.5 * 0.8_f64.powi((horizon + 1) as i32);
        assert_eq!(probs.len(), 2);
        assert!(probs.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
        assert!((probs[0] - expected_zero).abs() < 1e-10, "horizon {}", horizon + 1);
        assert!((probs[1] - (1.0 - expected_zero)).abs() < 1e-10);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }
    assert!(forecasts[0][0] > forecasts[10][0]);
    assert!((forecasts[255][0] - 0.5).abs() < 1e-10);
}

#[test]
fn long_observation_history_changes_forecast_without_underflow() {
    let mut predictor = persistent_binary_model();
    // 512 observations: switching from zeros to ones must reverse the forecast.
    // Forward filtering is normalized after every observation.
    let mut history = vec![0; 256];
    history.extend(std::iter::repeat_n(1, 256));

    let probs = predictor.forecast_probabilities(&history, 128).unwrap();
    assert_eq!(probs.len(), 128);
    assert!((probs[0][1] - 0.9).abs() < 1e-10);
    assert!((probs[1][1] - 0.82).abs() < 1e-10);
    assert!((probs[127].iter().sum::<f64>() - 1.0).abs() < 1e-10);

    predictor.set_history(&history);
    let mut output = [f32::NAN; 128];
    assert_eq!(predictor.predict(&mut output), TransformResult::Ok);
    assert!(output.iter().all(|&symbol| symbol == 1.0 || symbol == 0.0));
    assert_eq!(output[0], 1.0);
}
