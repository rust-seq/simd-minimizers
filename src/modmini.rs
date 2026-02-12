//! fastmod support for modmini

use super::S;

/// FastMod implementation that works for 8bit values.
/// Needs 2 10-cycle multiplications.
pub struct FastModU8 {
    /// the modulus
    d: S,
    /// the multiplication constant
    m: S,
}

impl FastModU8 {
    #[inline(always)]
    pub fn new(d: u8) -> Self {
        assert!(d > 0);
        Self {
            d: S::splat(d as u32),
            m: S::splat(((u16::MAX / d as u16) as u32).wrapping_add(1)),
        }
    }

    /// Return x%d in each lane when `x < 2^16`.
    #[inline(always)]
    pub fn reduce(&self, x: S) -> S {
        let lowbits = (self.m * x) & S::splat(u16::MAX as u32);
        (lowbits * self.d) >> 16
    }
}

#[test]
fn test_fastmod() {
    for d in 1..256 {
        let fm = FastModU8::new(d as u8);
        for x in 0..u8::MAX as usize {
            assert_eq!(
                fm.reduce(S::splat(x as u32)).as_array_ref()[0] as usize,
                x % d,
                "failure for d = {d}, x = {x}",
            );
        }
    }
}

#[inline(always)]
pub fn simple_mod(x: S, w: u8) -> S {
    x.cmp_gt(S::splat((w - 1) as u32))
        .blend(x - S::splat(w as u32), x)
}
