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

        pub fn get_data(&self) -> &[i32; 4] {
            &self.data
        }

        pub fn add(&mut self, other: &Vector) {
            unsafe {
                let va = vld1q_s32(self.data.as_ptr());
                let vb = vld1q_s32(other.data.as_ptr());
                let vc = vaddq_s32(va, vb);
                vst1q_s32(self.data.as_mut_ptr(), vc);
            }
        }

        pub fn scalar_add(&mut self, scalar: i32) {
            unsafe {
                let va = vld1q_s32(self.data.as_ptr());
                let vscalar = vdupq_n_s32(scalar);
                let vadd = vaddq_s32(va, vscalar);
                vst1q_s32(self.data.as_mut_ptr(), vadd);
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

        pub fn scalar_subtract(&mut self, scalar: i32) {
            unsafe {
                let va = vld1q_s32(self.data.as_ptr());
                let vscalar = vdupq_n_s32(scalar);
                let vsub = vsubq_s32(va, vscalar);
                vst1q_s32(self.data.as_mut_ptr(), vsub);
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

        pub fn abs_difference(&mut self, other: &Vector) {
            unsafe {
                let va = vld1q_s32(self.data.as_ptr());
                let vb = vld1q_s32(other.data.as_ptr());
                let vdiff = vabdq_s32(va, vb);
                vst1q_s32(self.data.as_mut_ptr(), vdiff);
            }
        }        

        pub fn sum(&self) -> i32 {
            unsafe {
                let va = vld1q_s32(self.data.as_ptr());
                let vadd = vaddvq_s32(va);
                vadd
            }
        }        

        pub fn sqrt(&mut self) {
            unsafe {
                let va = vreinterpretq_f32_s32(vld1q_s32(self.data.as_ptr()));
                let vsqrt = vsqrtq_f32(va);
                let vresult = vreinterpretq_s32_f32(vsqrt);
                vst1q_s32(self.data.as_mut_ptr(), vresult);
            }
        }              

        pub fn negate(&mut self) {
            unsafe {
                let va = vld1q_s32(self.data.as_ptr());
                let vneg = vnegq_s32(va);
                vst1q_s32(self.data.as_mut_ptr(), vneg);
            }
        }   

        pub fn concat(&self, other: &Vector) -> Vector {
            let mut result = Vector::new(0, 0, 0, 0);
            result.data[..2].copy_from_slice(&self.data);
            result.data[2..].copy_from_slice(&other.data);
            result
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use neon::Vector;