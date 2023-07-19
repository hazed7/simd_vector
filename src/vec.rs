#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    pub struct Vector {
        data: [i32; 4],
    }

    impl Vector {
        pub fn new(a: i32, b: i32, c: i32, d: i32) -> Self {
            Vector {
                data: [a, b, c, d],
            }
        }

        pub fn add(&mut self, other: &Vector) {
            unsafe {
                let va = vld1q_s32(self.data.as_ptr());
                let vb = vld1q_s32(other.data.as_ptr());
                let vc = vaddq_s32(va, vb);
                vst1q_s32(self.data.as_mut_ptr(), vc);
            }
        }

        pub fn subtract(&mut self, other: &Vector) {
            unsafe {
                let va = vld1q_s32(self.data.as_ptr());
                let vb = vld1q_s32(other.data.as_ptr());
                let vc = vsubq_s32(va, vb);
                vst1q_s32(self.data.as_mut_ptr(), vc);
            }
        }

        pub fn multiply(&mut self, other: &Vector) {
            unsafe {
                let va = vld1q_s32(self.data.as_ptr());
                let vb = vld1q_s32(other.data.as_ptr());
                let vc = vmulq_s32(va, vb);
                vst1q_s32(self.data.as_mut_ptr(), vc);
            }
        }

        pub fn scalar_multiply(&mut self, scalar: i32) {
            unsafe {
                let va = vld1q_s32(self.data.as_ptr());
                let vscalar = vdupq_n_s32(scalar);
                let vmul = vmulq_s32(va, vscalar);
                vst1q_s32(self.data.as_mut_ptr(), vmul);
            }
        }

        pub fn maximum(&mut self, other: &Vector) {
            unsafe {
                let va = vld1q_s32(self.data.as_ptr());
                let vb = vld1q_s32(other.data.as_ptr());
                let vc = vmaxq_s32(va, vb);
                vst1q_s32(self.data.as_mut_ptr(), vc);
            }
        }

        pub fn minimum(&mut self, other: &Vector) {
            unsafe {
                let va = vld1q_s32(self.data.as_ptr());
                let vb = vld1q_s32(other.data.as_ptr());
                let vc = vminq_s32(va, vb);
                vst1q_s32(self.data.as_mut_ptr(), vc);
            }
        }

        pub fn dot_product(&self, other: &Vector) -> i32 {
            unsafe {
                let va = vld1q_s32(self.data.as_ptr());
                let vb = vld1q_s32(other.data.as_ptr());
                let vmul = vmulq_s32(va, vb);
                let vadd = vpaddq_s32(vmul, vmul);
                let vadd = vaddq_s32(vadd, vrev64q_s32(vadd));
                vgetq_lane_s32(vadd, 0)
            }
        }    

        pub fn abs(&mut self) {
            unsafe {
                let va = vld1q_s32(self.data.as_ptr());
                let vabs = vabsq_s32(va);
                vst1q_s32(self.data.as_mut_ptr(), vabs);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use neon::Vector;