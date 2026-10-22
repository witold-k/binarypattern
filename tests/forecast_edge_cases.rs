use binarypattern::{HmmmPredict, PredictFollowF32, TransformResult};
use hmmm::HMM;
use ndarray::array;
use rand::{rngs::StdRng, SeedableRng};

fn predictor() -> HmmmPredict<HMM> {
    let hmm = HMM::new(
        array![[0.9, 0.1], [0.1, 0.9]],
        array![[1.0, 0.0], [0.0, 1.0]],
        array![0.5, 0.5],
    );
    HmmmPredict::new(hmm, StdRng::seed_from_u64(42))
}

#[test]
fn empty_history_has_unconditional_distribution() {
    let predictor = predictor();
    let probs = predictor.forecast_probabilities(&[], 3).unwrap();
    assert_eq!(probs.len(), 3);
    for step in probs {
        assert_eq!(step.len(), 2);
        assert!((step[0] - 0.5).abs() < 1e-10);
        assert!((step[1] - 0.5).abs() < 1e-10);
    }
}

#[test]
fn longer_history_changes_the_posterior() {
    let predictor = predictor();
    let once = predictor.forecast_probabilities(&[0], 1).unwrap();
    let twice = predictor.forecast_probabilities(&[0, 0], 1).unwrap();
    assert!(twice[0][0] > 0.5);
    assert!((once[0][0] - twice[0][0]).abs() < 1e-10);
}

#[test]
fn invalid_history_is_rejected_and_predict_reports_error() {
    let mut predictor = predictor();
    assert!(predictor.forecast_probabilities(&[2], 1).is_err());
    predictor.set_history(&[2]);
    let mut output = [f32::NAN; 2];
    assert_eq!(predictor.predict(&mut output), TransformResult::Error(1));
    assert!(output.iter().all(|x| x.is_nan()));
}

#[test]
fn empty_forecast_and_empty_output_are_supported() {
    let mut predictor = predictor();
    assert!(predictor.forecast_probabilities(&[0], 0).unwrap().is_empty());
    predictor.set_history(&[0]);
    let mut output = [];
    assert_eq!(predictor.predict(&mut output), TransformResult::Ok);
}
