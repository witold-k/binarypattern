use binarypattern::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_last_n() {

        let vec = [
            0.0, 1.0, 0.0, 0.0,  1.0, 1.0, 1.0, 1.0,  0.0, 1.0, 1.0, 1.0,  1.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 1.0,  0.0, 1.0, 1.0, 0.0,  0.0, 1.0, 0.0, 1.0,  0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,  1.0, 0.0, 0.0, 1.0,  1.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 1.0, 1.0,  1.0, 0.0, 1.0, 0.0,  1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 1.0,
            0.0
        ];

        let mut model = HmmmModel::<f64>::new(10, 3, 1234);
        let mut train = model.train(vec.into_iter().take(48));
        let mut out = [ 0.0, 0.0 ];
        let _ = train.predict(&mut out);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        println!("{:?}", out);
    }
}
