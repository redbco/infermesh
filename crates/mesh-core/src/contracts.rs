pub struct Labels {
    pub model: String,
    pub revision: String,
    pub quant: Option<String>,
    pub runtime: String,        // "triton" | "vllm" | "tgi" | …
    pub node: String,
    pub gpu_uuid: Option<String>,
    pub mig_profile: Option<String>,
    pub tenant: Option<String>,
    pub zone: Option<String>,
}

#[derive(Clone, Copy)]
pub enum SloClass { Latency, Throughput }

pub trait RuntimeControl: Send + Sync {
    fn load(&self, model: &str, revision: &str) -> anyhow::Result<()>;
    fn unload(&self, model: &str, revision: &str) -> anyhow::Result<()>;
    fn warm(&self, model: &str, revision: &str) -> anyhow::Result<()>;
}

pub struct ModelState {
    pub queue_depth: u32,
    pub service_rate: f64,   // req/s or tokens/s
    pub p95_ms: u32,
    pub batch_fullness: f32, // 0..1
    pub loaded: bool,
}

pub struct GpuState {
    pub gpu_uuid: String,
    pub mig_profile: Option<String>,
    pub sm_utilization: f32, // 0..1
    pub vram_used_gb: f32,
    pub vram_total_gb: f32,
    pub ecc_fault: bool,
}