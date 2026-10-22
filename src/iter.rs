use crate::PredictReadIterator;

impl<T> PredictReadIterator for T where T:
      core::iter::Iterator
    + core::iter::DoubleEndedIterator
    + core::iter::ExactSizeIterator
    + Clone
{
}

