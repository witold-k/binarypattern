use binarypattern::*;
use hmmm::HMM;
use ndarray_013::array;
use rand::SeedableRng as _;

#[test]
fn forecast_depends_on_observed_history() {
    // Each hidden state emits one symbol; states persist with 90% probability.
    let hmm = HMM::new(
        array![[0.9, 0.1], [0.1, 0.9]],
        array![[1.0, 0.0], [0.0, 1.0]],
        array![0.5, 0.5],
    );
    let rng = rand::rngs::StdRng::seed_from_u64(1234);
    let mut predictor = HmmmPredict::<HMM>::new(hmm, rng);
    let after_zero = predictor.forecast_probabilities(&[0], 2).unwrap();
    let after_one = predictor.forecast_probabilities(&[1], 2).unwrap();
    assert!((after_zero[0][0] - 0.9).abs() < 1e-10);
    assert!((after_one[0][1] - 0.9).abs() < 1e-10);
    assert!((after_zero[1][0] - 0.82).abs() < 1e-10);
    assert!((after_zero[0].iter().sum::<f64>() - 1.0).abs() < 1e-10);
    assert!(predictor.forecast_probabilities(&[2], 1).is_err());
    predictor.set_history(&[1]);
    let mut out = [0.0; 2];
    assert_eq!(predictor.predict(&mut out), TransformResult::Ok);
    assert_eq!(out, [1.0, 1.0]);
}

#[test]
fn trained_model_can_forecast() {
    let sequence = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let mut model = HmmmModel::<f64>::new(2, 2, 1234);
    let predictor = model.train(sequence.into_iter());
    let probs = predictor.forecast_probabilities(&[0, 1], 2).unwrap();
    assert_eq!(probs.len(), 2);
    for p in probs {
        assert_eq!(p.len(), 2);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-8);
    }
}
