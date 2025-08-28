use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, LogNormal, Poisson, Zipf};
use serde::{Deserialize, Serialize};
use crate::engine::{Request, RequestType, RequestId};

/// Workload configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadConfig {
    /// Simulation duration in seconds
    pub duration_s: f64,
    /// Arrival process configuration
    pub arrival: ArrivalConfig,
    /// Request type mix
    pub mix: RequestMix,
    /// LLM-specific configuration
    pub llm: LlmConfig,
    /// Vision-specific configuration (optional)
    pub vision: Option<VisionConfig>,
    /// ASR-specific configuration (optional)
    pub asr: Option<AsrConfig>,
    /// Tenant configuration (optional)
    pub tenants: Option<TenantConfig>,
}

/// Arrival process configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ArrivalConfig {
    #[serde(rename = "poisson")]
    Poisson { rps: f64 },
    #[serde(rename = "mmpp")]
    MMPP {
        states: usize,
        rates_rps: Vec<f64>,
        dwell_s: Vec<f64>,
    },
}

/// Request type mix configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMix {
    pub llm: f64,
    pub vision: f64,
    pub asr: f64,
}

impl RequestMix {
    /// Validate that mix percentages sum to 1.0
    pub fn validate(&self) -> Result<(), String> {
        let total = self.llm + self.vision + self.asr;
        if (total - 1.0).abs() > 0.001 {
            Err(format!("Request mix must sum to 1.0, got {}", total))
        } else {
            Ok(())
        }
    }

    /// Sample a request type based on the mix
    pub fn sample_request_type(&self, rng: &mut SmallRng) -> RequestType {
        let r: f64 = rng.gen();
        if r < self.llm {
            RequestType::LLM
        } else if r < self.llm + self.vision {
            RequestType::Vision
        } else {
            RequestType::ASR
        }
    }
}

/// LLM workload configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub in_tokens: TokenDistribution,
    pub out_tokens: TokenDistribution,
}

/// Vision workload configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    pub image_size: ImageSizeDistribution,
    pub processing_complexity: ComplexityDistribution,
}

/// ASR workload configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrConfig {
    pub audio_duration_s: DurationDistribution,
    pub sample_rate: u32,
}

/// Token distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "dist")]
pub enum TokenDistribution {
    #[serde(rename = "lognormal")]
    LogNormal { mu: f64, sigma: f64 },
    #[serde(rename = "poisson")]
    Poisson { lambda: f64 },
    #[serde(rename = "constant")]
    Constant { value: u32 },
}

impl TokenDistribution {
    /// Sample token count from this distribution
    pub fn sample(&self, rng: &mut SmallRng) -> u32 {
        match self {
            TokenDistribution::LogNormal { mu, sigma } => {
                let log_normal = LogNormal::new(*mu, *sigma).unwrap();
                log_normal.sample(rng).round() as u32
            }
            TokenDistribution::Poisson { lambda } => {
                let poisson = Poisson::new(*lambda).unwrap();
                poisson.sample(rng) as u32
            }
            TokenDistribution::Constant { value } => *value,
        }
    }
}

/// Image size distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "dist")]
pub enum ImageSizeDistribution {
    #[serde(rename = "uniform")]
    Uniform { min_pixels: u32, max_pixels: u32 },
    #[serde(rename = "discrete")]
    Discrete { sizes: Vec<(u32, f64)> }, // (pixels, probability)
}

impl ImageSizeDistribution {
    pub fn sample(&self, rng: &mut SmallRng) -> u32 {
        match self {
            ImageSizeDistribution::Uniform { min_pixels, max_pixels } => {
                rng.gen_range(*min_pixels..*max_pixels)
            }
            ImageSizeDistribution::Discrete { sizes } => {
                let r: f64 = rng.gen();
                let mut cumulative = 0.0;
                for (pixels, prob) in sizes {
                    cumulative += prob;
                    if r <= cumulative {
                        return *pixels;
                    }
                }
                sizes.last().unwrap().0 // Fallback
            }
        }
    }
}

/// Processing complexity distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "dist")]
pub enum ComplexityDistribution {
    #[serde(rename = "uniform")]
    Uniform { min: f64, max: f64 },
    #[serde(rename = "constant")]
    Constant { value: f64 },
}

impl ComplexityDistribution {
    pub fn sample(&self, rng: &mut SmallRng) -> f64 {
        match self {
            ComplexityDistribution::Uniform { min, max } => rng.gen_range(*min..*max),
            ComplexityDistribution::Constant { value } => *value,
        }
    }
}

/// Audio duration distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "dist")]
pub enum DurationDistribution {
    #[serde(rename = "exponential")]
    Exponential { lambda: f64 },
    #[serde(rename = "uniform")]
    Uniform { min_s: f64, max_s: f64 },
}

impl DurationDistribution {
    pub fn sample(&self, rng: &mut SmallRng) -> f64 {
        match self {
            DurationDistribution::Exponential { lambda } => {
                let exp = rand_distr::Exp::new(*lambda).unwrap();
                exp.sample(rng)
            }
            DurationDistribution::Uniform { min_s, max_s } => rng.gen_range(*min_s..*max_s),
        }
    }
}

/// Tenant configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    pub skew: TenantSkew,
    pub count: usize,
}

/// Tenant skew configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TenantSkew {
    #[serde(rename = "zipf")]
    Zipf { s: f64 },
    #[serde(rename = "uniform")]
    Uniform,
}

impl TenantSkew {
    /// Sample a tenant ID based on the skew
    pub fn sample_tenant_id(&self, tenant_count: usize, rng: &mut SmallRng) -> String {
        match self {
            TenantSkew::Zipf { s } => {
                let zipf = Zipf::new(tenant_count as u64, *s).unwrap();
                let tenant_id = zipf.sample(rng) as u64;
                format!("tenant-{}", tenant_id)
            }
            TenantSkew::Uniform => {
                let tenant_id = rng.gen_range(1..=tenant_count);
                format!("tenant-{}", tenant_id)
            }
        }
    }
}

/// Markov-Modulated Poisson Process state
#[derive(Debug, Clone)]
struct MMPPState {
    current_state: usize,
    state_start_time: f64,
    rates: Vec<f64>,
    dwell_times: Vec<f64>,
}

impl MMPPState {
    fn new(rates: Vec<f64>, dwell_times: Vec<f64>) -> Self {
        Self {
            current_state: 0,
            state_start_time: 0.0,
            rates,
            dwell_times,
        }
    }

    fn update(&mut self, current_time: f64, _rng: &mut SmallRng) {
        let time_in_state = current_time - self.state_start_time;
        if time_in_state >= self.dwell_times[self.current_state] {
            // Transition to next state
            self.current_state = (self.current_state + 1) % self.rates.len();
            self.state_start_time = current_time;
        }
    }

    fn current_rate(&self) -> f64 {
        self.rates[self.current_state]
    }
}

/// Workload generator
#[derive(Clone)]
pub struct WorkloadGenerator {
    config: WorkloadConfig,
    rng: SmallRng,
    mmpp_state: Option<MMPPState>,
    next_request_id: RequestId,
}

impl WorkloadGenerator {
    /// Create a new workload generator
    pub fn new(config: WorkloadConfig, seed: u64) -> Result<Self, String> {
        config.mix.validate()?;

        let mmpp_state = match &config.arrival {
            ArrivalConfig::MMPP { states, rates_rps, dwell_s } => {
                if rates_rps.len() != *states || dwell_s.len() != *states {
                    return Err("MMPP rates and dwell times must match state count".to_string());
                }
                Some(MMPPState::new(rates_rps.clone(), dwell_s.clone()))
            }
            _ => None,
        };

        Ok(Self {
            config,
            rng: SmallRng::seed_from_u64(seed),
            mmpp_state,
            next_request_id: 1,
        })
    }

    /// Generate the next request arrival time
    pub fn next_arrival_time(&mut self, current_time: f64) -> f64 {
        match &self.config.arrival {
            ArrivalConfig::Poisson { rps } => {
                let exp = rand_distr::Exp::new(*rps).unwrap();
                let inter_arrival_time = exp.sample(&mut self.rng);
                current_time + inter_arrival_time * 1000.0 // Convert to milliseconds
            }
            ArrivalConfig::MMPP { .. } => {
                if let Some(ref mut mmpp) = self.mmpp_state {
                    mmpp.update(current_time, &mut self.rng);
                    let rate = mmpp.current_rate();
                    let exp = rand_distr::Exp::new(rate).unwrap();
                    let inter_arrival_time = exp.sample(&mut self.rng);
                    current_time + inter_arrival_time * 1000.0 // Convert to milliseconds
                } else {
                    current_time + 1000.0 // Fallback
                }
            }
        }
    }

    /// Generate a new request
    pub fn generate_request(&mut self, arrival_time: f64) -> Request {
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let request_type = self.config.mix.sample_request_type(&mut self.rng);
        
        let (input_tokens, expected_output_tokens) = match request_type {
            RequestType::LLM => {
                let input = self.config.llm.in_tokens.sample(&mut self.rng);
                let output = self.config.llm.out_tokens.sample(&mut self.rng);
                (input, output)
            }
            RequestType::Vision => {
                // Convert image processing to token-equivalent
                if let Some(ref vision_config) = self.config.vision {
                    let pixels = vision_config.image_size.sample(&mut self.rng);
                    let complexity = vision_config.processing_complexity.sample(&mut self.rng);
                    let equivalent_tokens = ((pixels as f64 / 1000.0) * complexity) as u32;
                    (equivalent_tokens, equivalent_tokens / 4) // Assume 4:1 input:output ratio
                } else {
                    (1000, 250) // Default values
                }
            }
            RequestType::ASR => {
                // Convert audio processing to token-equivalent
                if let Some(ref asr_config) = self.config.asr {
                    let duration = asr_config.audio_duration_s.sample(&mut self.rng);
                    let equivalent_tokens = (duration * 100.0) as u32; // ~100 tokens per second of audio
                    (equivalent_tokens, equivalent_tokens / 2) // Assume 2:1 input:output ratio
                } else {
                    (500, 250) // Default values
                }
            }
        };

        let tenant_id = if let Some(ref tenant_config) = self.config.tenants {
            tenant_config.skew.sample_tenant_id(tenant_config.count, &mut self.rng)
        } else {
            "default-tenant".to_string()
        };

        Request {
            id: request_id,
            request_type,
            tenant_id,
            model_id: self.select_model_for_request_type(request_type),
            arrival_time,
            input_tokens,
            expected_output_tokens,
            sla_ms: self.calculate_sla(request_type, input_tokens + expected_output_tokens),
        }
    }

    /// Select appropriate model for request type
    fn select_model_for_request_type(&mut self, _request_type: RequestType) -> String {
        match _request_type {
            RequestType::LLM => {
                // TODO: Implement model selection logic
                "llama-70b".to_string()
            }
            RequestType::Vision => "clip-vit-large".to_string(),
            RequestType::ASR => "whisper-large".to_string(),
        }
    }

    /// Calculate SLA for a request
    fn calculate_sla(&self, request_type: RequestType, total_tokens: u32) -> Option<f64> {
        // Simple SLA calculation based on request type and size
        let base_sla_ms = match request_type {
            RequestType::LLM => 5000.0, // 5 seconds for LLM
            RequestType::Vision => 2000.0, // 2 seconds for vision
            RequestType::ASR => 3000.0, // 3 seconds for ASR
        };

        // Scale by token count (larger requests get more time)
        let token_factor = (total_tokens as f64 / 1000.0).max(1.0);
        Some(base_sla_ms * token_factor)
    }

    /// Check if we should continue generating requests
    pub fn should_continue(&self, current_time_ms: f64) -> bool {
        current_time_ms < self.config.duration_s * 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> WorkloadConfig {
        WorkloadConfig {
            duration_s: 300.0,
            arrival: ArrivalConfig::Poisson { rps: 100.0 },
            mix: RequestMix {
                llm: 0.7,
                vision: 0.2,
                asr: 0.1,
            },
            llm: LlmConfig {
                in_tokens: TokenDistribution::LogNormal { mu: 3.8, sigma: 0.6 },
                out_tokens: TokenDistribution::LogNormal { mu: 4.6, sigma: 0.7 },
            },
            vision: Some(VisionConfig {
                image_size: ImageSizeDistribution::Uniform {
                    min_pixels: 224 * 224,
                    max_pixels: 1024 * 1024,
                },
                processing_complexity: ComplexityDistribution::Constant { value: 1.0 },
            }),
            asr: Some(AsrConfig {
                audio_duration_s: DurationDistribution::Uniform { min_s: 1.0, max_s: 30.0 },
                sample_rate: 16000,
            }),
            tenants: Some(TenantConfig {
                skew: TenantSkew::Zipf { s: 1.1 },
                count: 100,
            }),
        }
    }

    #[test]
    fn test_request_mix_validation() {
        let valid_mix = RequestMix {
            llm: 0.7,
            vision: 0.2,
            asr: 0.1,
        };
        assert!(valid_mix.validate().is_ok());

        let invalid_mix = RequestMix {
            llm: 0.7,
            vision: 0.2,
            asr: 0.2, // Sum > 1.0
        };
        assert!(invalid_mix.validate().is_err());
    }

    #[test]
    fn test_token_distribution_sampling() {
        let mut rng = SmallRng::seed_from_u64(42);
        
        let log_normal = TokenDistribution::LogNormal { mu: 4.0, sigma: 0.5 };
        let poisson = TokenDistribution::Poisson { lambda: 100.0 };
        let constant = TokenDistribution::Constant { value: 500 };

        // Test sampling
        for _ in 0..10 {
            let ln_sample = log_normal.sample(&mut rng);
            let p_sample = poisson.sample(&mut rng);
            let c_sample = constant.sample(&mut rng);

            assert!(ln_sample > 0);
            assert!(p_sample > 0);
            assert_eq!(c_sample, 500);
        }
    }

    #[test]
    fn test_workload_generator() {
        let config = create_test_config();
        let mut generator = WorkloadGenerator::new(config, 42).unwrap();

        // Generate some requests
        let mut current_time = 0.0;
        let mut requests = Vec::new();

        for _ in 0..10 {
            let arrival_time = generator.next_arrival_time(current_time);
            let request = generator.generate_request(arrival_time);
            requests.push(request);
            current_time = arrival_time;
        }

        assert_eq!(requests.len(), 10);
        
        // Check that arrival times are increasing
        for i in 1..requests.len() {
            assert!(requests[i].arrival_time >= requests[i-1].arrival_time);
        }

        // Check that we have different request types
        let llm_count = requests.iter().filter(|r| r.request_type == RequestType::LLM).count();
        let vision_count = requests.iter().filter(|r| r.request_type == RequestType::Vision).count();
        let asr_count = requests.iter().filter(|r| r.request_type == RequestType::ASR).count();

        assert!(llm_count > 0); // Should have some LLM requests given 70% probability
        assert_eq!(llm_count + vision_count + asr_count, 10);
    }

    #[test]
    fn test_mmpp_configuration() {
        let mmpp_config = WorkloadConfig {
            duration_s: 300.0,
            arrival: ArrivalConfig::MMPP {
                states: 3,
                rates_rps: vec![100.0, 500.0, 1000.0],
                dwell_s: vec![30.0, 10.0, 5.0],
            },
            mix: RequestMix { llm: 1.0, vision: 0.0, asr: 0.0 },
            llm: LlmConfig {
                in_tokens: TokenDistribution::Constant { value: 100 },
                out_tokens: TokenDistribution::Constant { value: 200 },
            },
            vision: None,
            asr: None,
            tenants: None,
        };

        let generator = WorkloadGenerator::new(mmpp_config, 42);
        assert!(generator.is_ok());
    }
}
