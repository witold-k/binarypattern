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

        let mut model = HmmmModel::<f64>::new(2, 3, 1234);
        let mut train = model.train(vec.into_iter().take(48));
        let mut out = [ 0.0, 0.0 ];
        let _ = train.predict(&mut out);
        assert!(out.iter().all(|&x| x == 0.0 || x == 1.0));
        println!("{:?}", out);
    }
}
