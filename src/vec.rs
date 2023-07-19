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
    }
}

#[cfg(target_arch = "aarch64")]
pub use neon::Vector;