use binarypattern::{HmmmModel, PredictFollowF32, TrainF32, TransformResult};

fn predict_sequence(data: &[f32], seed: u64, size: usize) -> Vec<f32> {
    let mut model = HmmmModel::<f32>::new(2, 2, seed);
    let mut predictor = model.train(data.iter().copied());
    let mut output = vec![f32::NAN; size];
    assert_eq!(predictor.predict(&mut output), TransformResult::Ok);
    output
}

#[test]
fn predictions_are_valid_binary_observations() {
    // Both observation symbols occur repeatedly in the training sequence.
    let training: Vec<f32> = (0..120).map(|i| (i % 2) as f32).collect();
    let output = predict_sequence(&training, 42, 64);
    assert_eq!(output.len(), 64);
    assert!(
        output.iter().all(|&x| x == 0.0 || x == 1.0),
        "predictions must belong to the configured observation alphabet: {output:?}"
    );
}

#[test]
fn predictions_are_reproducible_with_the_same_seed() {
    let training: Vec<f32> = (0..120).map(|i| ((i / 3) % 2) as f32).collect();
    let first = predict_sequence(&training, 1234, 32);
    let second = predict_sequence(&training, 1234, 32);
    assert_eq!(first, second, "same training data and seed should reproduce samples");
}

#[test]
fn prediction_populates_the_full_output_buffer() {
    let training: Vec<f32> = (0..120).map(|i| (i % 2) as f32).collect();
    let output = predict_sequence(&training, 17, 16);
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn prediction_supports_an_empty_output_buffer() {
    let training: Vec<f32> = (0..120).map(|i| (i % 2) as f32).collect();
    let output = predict_sequence(&training, 17, 0);
    assert!(output.is_empty());
}
