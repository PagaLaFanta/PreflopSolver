//! PRNG mínimo. SplitMix64 pasa BigCrush, tiene estado de 8 bytes (cabe en un registro,
//! se puede tener uno por hilo sin coste) y es suficiente tanto para Monte Carlo ICM como
//! para el redondeo estocástico de la cuantización de regrets.

#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    #[inline(always)]
    pub const fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniforme en `(0, 1]` con 53 bits de mantisa. Nunca 0 → `-ln(u)` es finito.
    #[inline(always)]
    pub fn next_f64_open0(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniforme en `[0, 1)` con 24 bits: exactamente la resolución que necesita el
    /// redondeo estocástico a enteros de 16 bits.
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u32 << 24) as f32)
    }
}
