use super::*;

/// This implementation is mostly copied from Daniel Liu's gist:
/// https://gist.github.com/Daniel-Liu-c0deb0t/7078ebca04569068f15507aa856be6e8
///
/// Modifications:
/// 1. Add a wrapper type to implement our Minimizer trait.
/// 2. Always return the leftmost minimum, instead of using robust winnowing.
pub struct RescanDaniel {
    pub k: usize,
    pub w: usize,
}

impl Minimizer for RescanDaniel {
    fn window_minimizers(&mut self, text: &[u8]) -> Vec<usize> {
        let mut minimizers = Vec::new();
        minimizers_callback(text, self.w + self.k - 1, self.k, 0, |idx, _min| {
            minimizers.push(idx);
        });
        minimizers
    }
}

/// ntHash constants.
static LUT: [u64; 128] = {
    let mut l = [0u64; 128];
    l[b'A' as usize] = 0x3c8bfbb395c60474u64;
    l[b'C' as usize] = 0x3193c18562a02b4cu64;
    l[b'G' as usize] = 0x20323ed082572324u64;
    l[b'T' as usize] = 0x295549f54be24456u64;
    l
};

/// Robust winnowing.
fn minimizers_callback(s: &[u8], l: usize, k: usize, prev: u64, mut f: impl FnMut(usize, u64)) {
    let mut min = 0;
    let mut min_idx = 0;
    let mut curr = 0;

    for (i, win) in s.windows(l).enumerate() {
        if i == 0 || i > min_idx {
            let (m_idx, m, c) = minimum(win, k, prev);
            min_idx = i + m_idx;
            min = m;
            curr = c;
            f(min_idx, min);
        } else {
            curr = curr.rotate_left(1)
                ^ LUT[win[l - 1 - k] as usize].rotate_left(k as u32)
                ^ LUT[win[l - 1] as usize];
            let h = prev ^ curr;

            if h < min {
                min_idx = i + l - k;
                min = h;
            }
            f(min_idx, min);
        }
    }
}

/// Get the rightmost minimum kmer.
fn minimum(s: &[u8], k: usize, prev: u64) -> (usize, u64, u64) {
    let mut curr = 0;

    for (i, &b) in s[..k].iter().enumerate() {
        curr ^= LUT[b as usize].rotate_left((k - 1 - i) as u32);
    }

    let mut min = prev ^ curr;
    let mut min_idx = 0;

    for (i, &b) in s[k..].iter().enumerate() {
        curr = curr.rotate_left(1) ^ LUT[s[i] as usize].rotate_left(k as u32) ^ LUT[b as usize];
        let h = prev ^ curr;

        // This was changed from <= to < to ensure the leftmost minimum is returned.
        if h < min {
            min = h;
            min_idx = i + 1;
        }
    }

    (min_idx, min, curr)
}
