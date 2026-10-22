use hmmm::HMM;
use rand::SeedableRng as _;
use rand::rngs::StdRng;

use crate::*;


impl<T> HmmmModel<T> {

    #[inline(always)]
    pub fn new(observations_n: usize, hidden_states_k: usize, seed: u64) -> Self {
        HmmmModel {
            marker: Default::default(),
            observations_n,
            hidden_states_k,
            rng: StdRng::seed_from_u64(seed)
        }
    }
}

impl<T> HmmmPredict<T> {

    #[inline(always)]
    pub fn new(hmm: HMM, rng: StdRng) -> Self {
        HmmmPredict {
            marker: Default::default(),
            hmm, rng
        }
    }
}


impl<T> TrainF32<HmmmPredict<HMM>> for HmmmModel<T> {

    fn train<'a, RI>(
        &mut self,
        ri: RI
    ) -> HmmmPredict<HMM>
    where
        RI: PredictReadIterator<Item = f32>
    {
        let data = ri.map(|x| x as usize).collect();
        let hmm = HMM::train(&data, self.observations_n, self.hidden_states_k, &mut self.rng);
        HmmmPredict::<HMM>::new(hmm, self.rng.clone())
    }
}

impl<T> PredictFollowF32<HMM> for HmmmPredict<T> {
    fn predict(
        &mut self,
        out: &mut [f32]
    ) -> TransformResult
    {
        let sampled_i = self.hmm.sampler(&mut self.rng)
            .map(|sample| sample.y)
            .take(out.len());

        let out_i = out.iter_mut();
        for (we, re) in out_i.zip(sampled_i) { *we = re as f32; }

        TransformResult::Ok
    }

}

