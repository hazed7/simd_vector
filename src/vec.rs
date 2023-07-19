#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    pub struct Vector {
        data: [f32; 4],
    }

    impl Vector {
        pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
            Self {
                data: [a, b, c, d],
            }
        }

        pub fn zero() -> Self {
            Self {
                data: [0.0, 0.0, 0.0, 0.0],
            }
        }

        pub fn unit_x() -> Self {
            Vector {
                data: [1.0, 0.0, 0.0, 0.0],
            }
        }

        pub fn unit_y() -> Self {
            Vector {
                data: [0.0, 1.0, 0.0, 0.0],
            }
        }

        pub fn unit_z() -> Self {
            Vector {
                data: [0.0, 0.0, 1.0, 0.0],
            }
        }

        pub fn unit_w() -> Self {
            Vector {
                data: [0.0, 0.0, 0.0, 1.0],
            }
        }

        pub fn get_data(&self) -> &[f32; 4] {
            &self.data
        }

        pub fn add(&mut self, other: &Vector) {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vb = vld1q_f32(other.data.as_ptr());
                let vc = vaddq_f32(va, vb);
                vst1q_f32(self.data.as_mut_ptr(), vc);
            }
        }

        pub fn scalar_add(&mut self, scalar: f32) {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vscalar = vdupq_n_f32(scalar);
                let vadd = vaddq_f32(va, vscalar);
                vst1q_f32(self.data.as_mut_ptr(), vadd);
            }
        }

        pub fn subtract(&mut self, other: &Vector) {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vb = vld1q_f32(other.data.as_ptr());
                let vc = vsubq_f32(va, vb);
                vst1q_f32(self.data.as_mut_ptr(), vc);
            }
        }

        pub fn scalar_subtract(&mut self, scalar: f32) {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vscalar = vdupq_n_f32(scalar);
                let vsub = vsubq_f32(va, vscalar);
                vst1q_f32(self.data.as_mut_ptr(), vsub);
            }
        }

        pub fn multiply(&mut self, other: &Vector) {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vb = vld1q_f32(other.data.as_ptr());
                let vc = vmulq_f32(va, vb);
                vst1q_f32(self.data.as_mut_ptr(), vc);
            }
        }

        pub fn scalar_multiply(&mut self, scalar: f32) {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vscalar = vdupq_n_f32(scalar);
                let vmul = vmulq_f32(va, vscalar);
                vst1q_f32(self.data.as_mut_ptr(), vmul);
            }
        }

        pub fn maximum(&mut self, other: &Vector) {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vb = vld1q_f32(other.data.as_ptr());
                let vc = vmaxq_f32(va, vb);
                vst1q_f32(self.data.as_mut_ptr(), vc);
            }
        }

        pub fn minimum(&mut self, other: &Vector) {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vb = vld1q_f32(other.data.as_ptr());
                let vc = vminq_f32(va, vb);
                vst1q_f32(self.data.as_mut_ptr(), vc);
            }
        }

        pub fn dot_product(&self, other: &Vector) -> f32 {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vb = vld1q_f32(other.data.as_ptr());
                let vmul = vmulq_f32(va, vb);
                let vadd = vpaddq_f32(vmul, vmul);
                let vadd = vaddq_f32(vadd, vrev64q_f32(vadd));
                vgetq_lane_f32(vadd, 0)
            }
        }

        pub fn abs(&mut self) {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vabs = vabsq_f32(va);
                vst1q_f32(self.data.as_mut_ptr(), vabs);
            }
        }

        pub fn abs_difference(&mut self, other: &Vector) {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vb = vld1q_f32(other.data.as_ptr());
                let vdiff = vabdq_f32(va, vb);
                vst1q_f32(self.data.as_mut_ptr(), vdiff);
            }
        }

        pub fn sum(&self) -> f32 {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vadd = vaddvq_f32(va);
                vadd
            }
        }

        pub fn sqrt(&mut self) {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vsqrt = vsqrtq_f32(va);
                vst1q_f32(self.data.as_mut_ptr(), vsqrt);
            }
        }

        pub fn negate(&mut self) {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vneg = vnegq_f32(va);
                vst1q_f32(self.data.as_mut_ptr(), vneg);
            }
        }

        pub fn concat(&self, other: &Vector) -> Vector {
            let mut result = Vector::new(0.0, 0.0, 0.0, 0.0);
            result.data[..2].copy_from_slice(&self.data);
            result.data[2..].copy_from_slice(&other.data);
            result
        }

        pub fn clamp(&mut self, min: f32, max: f32) {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let vmin = vdupq_n_f32(min);
                let vmax = vdupq_n_f32(max);

                let vc = vmaxq_f32(vmin, va);
                let vc = vminq_f32(vc, vmax);

                vst1q_f32(self.data.as_mut_ptr(), vc);
            }
        }

        pub fn magnitude(&self) -> f32 {
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let va_float = va;
                let vmul = vmulq_f32(va_float, va_float);
                let vsum = vaddvq_f32(vmul);
                let vsqrt = vsqrtq_f32(vdupq_n_f32(vsum));
                vgetq_lane_f32(vsqrt, 0)
            }
        }

        pub fn normalize(&mut self) {
            let magnitude = self.magnitude();
            unsafe {
                let va = vld1q_f32(self.data.as_ptr());
                let va_float = va;
                let vrecip_mag = vrecpeq_f32(vdupq_n_f32(magnitude));
                let vnorm = vmulq_f32(va_float, vrecip_mag);
                let vnorm_int = vreinterpretq_s32_f32(vnorm);
                vst1q_f32(self.data.as_mut_ptr(), vreinterpretq_f32_s32(vnorm_int));
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use neon::Vector;