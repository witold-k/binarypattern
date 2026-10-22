use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{ Add, AddAssign, Sub, SubAssign, Mul, Div };
use serde::{Deserialize, Serialize};
use num_traits::sign;
use num_traits::cast::{ AsPrimitive, FromPrimitive };
use hmmm::HMM;

pub mod hmmmpredict;
pub mod iter;
pub mod number;

// ---------------------------------------------------------------------------

#[derive(Copy, Clone, PartialOrd, PartialEq, Hash, Deserialize, Serialize, Debug)]
pub enum TransformResult {
    Ok,
    Error(u32)
}

// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct HmmmModel<T> {
    marker: PhantomData<T>,
    observations_n:  usize,
    hidden_states_k: usize,
    rng:    rand_07::rngs::StdRng
}

pub struct HmmmPredict<T> {
    marker: PhantomData<T>,
    hmm:    HMM,
    history: Vec<usize>
}

// ---------------------------------------------------------------------------

pub trait PredictReadIterator:
      core::iter::Iterator
    + core::iter::DoubleEndedIterator
    + core::iter::ExactSizeIterator
    + Clone
{
}

pub trait Number<ItemIn, ItemOut = ItemIn>:
    Copy + Clone +
    PartialOrd<ItemIn> + PartialEq<ItemIn> +
    FromPrimitive +
    AsPrimitive<u8> + AsPrimitive<i8> +
    AsPrimitive<usize> + AsPrimitive<u32> + AsPrimitive<i32> +
    AsPrimitive<f32> + AsPrimitive<f64> +
    AddAssign<ItemIn> + SubAssign<ItemIn> +
    sign::Signed +
    Add<ItemIn, Output = ItemOut> + Sub<ItemIn, Output = ItemOut> +
    Mul<ItemIn, Output = ItemOut> + Div<ItemIn, Output = ItemOut> +
    Debug
{}

// ---------------------------------------------------------------------------

pub trait Predict<ITEMOUT> {
    fn transform<'a, RI, ITEMIN: 'a>(
        &mut self,
        ri: RI,
        out: &mut [ITEMOUT],
        default_in: ITEMIN
    ) -> TransformResult
    where
        RI: PredictReadIterator<Item = &'a ITEMIN>,
        ITEMOUT: Number<ITEMOUT>
    ;
}

pub trait TrainF32<MODEL> {
    fn train<RI>(
        &mut self,
        ri: RI
    ) -> MODEL
    where
        RI: PredictReadIterator<Item = f32>
    ;
}

pub trait PredictF32<MODEL> {
    fn predict<RI>(
        &mut self,
        ri: RI,
        out: &mut [f32]
    ) -> TransformResult
    where
        RI: PredictReadIterator<Item = f32>
    ;
}

pub trait PredictFollowF32<MODEL> {
    fn predict(
        &mut self,
        out: &mut [f32]
    ) -> TransformResult;
}

