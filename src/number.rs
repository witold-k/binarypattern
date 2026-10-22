use std::fmt::Debug;
use std::ops::{ Add, AddAssign, Sub, SubAssign, Mul, Div };
use num_traits::cast::{ AsPrimitive, FromPrimitive };
use num_traits::sign;

use crate::Number;

impl<ItemIn, ItemOut, T> Number<ItemIn, ItemOut> for T
where T:
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

