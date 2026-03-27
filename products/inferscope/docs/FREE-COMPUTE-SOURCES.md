# Free and Low-Cost GPU Compute for InferScope Benchmarking

## Priority: Where to Run Kimi-K2.5 and GLM-5 Benchmarks

Both models require **8x H100/H200 (FP8)** or **4x B200 (NVFP4)** minimum. Free tiers with these GPUs are extremely limited — most paths require credits programs or partnerships.

---

## Tier 1: Free Credits Programs (Best for Benchmarking)

### NVIDIA DGX Cloud Benchmarking
- **URL**: https://developer.nvidia.com/dgx-cloud/benchmarking
- **What**: Pre-built benchmarking recipes, access to published performance results at no cost
- **GPUs**: H100, H200, B200, GB200
- **How to use**: Run InferScope experiments using NVIDIA's containerized benchmark templates
- **Best for**: Publishing validated results that NVIDIA can reference

### AWS Activate (Startups)
- **Credits**: $1,000 – $100,000 depending on accelerator tier
- **Validity**: 1-2 years
- **GPUs**: P5 (H100), P5en (H200) instances
- **URL**: https://aws.amazon.com/activate/
- **Best for**: Extended benchmark campaigns across multiple model/GPU combinations
- **Requirement**: Must be in a recognized accelerator program

### Google Cloud (New Account)
- **Credits**: $300 for 90 days
- **GPUs**: A100 40GB (~30-40 hours), T4 (~100 hours)
- **URL**: https://console.cloud.google.com
- **Limitation**: No H100/H200 on free tier. Useful for A100 baseline comparisons only.

### Microsoft Azure (New Account / Students)
- **Credits**: $200 for 30 days (new users), $100/year (students)
- **GPUs**: NCv3 (V100), NDv2 (A100) series
- **URL**: https://azure.microsoft.com/en-us/free/

---

## Tier 2: Low-Cost Spot/On-Demand (Best ROI)

### RunPod
- **Signup credits**: $5-10 new user bonus
- **H100 spot**: ~$2.69/hr
- **H200 spot**: Often available
- **Cold start**: 200ms (FlashBoot)
- **Billing**: Per-second
- **URL**: https://www.runpod.io
- **InferScope integration**: Use SkyPilot YAML → RunPod backend

### Lambda Labs
- **A100**: $1.10/hr
- **H100**: ~$2.49/hr
- **URL**: https://lambdalabs.com
- **Academic program**: Free GPU time for published research
- **InferScope integration**: Direct SSH or SkyPilot backend

### GMI Cloud
- **H100**: $2.10/hr
- **H200**: $2.50/hr
- **URL**: https://www.gmicloud.ai
- **Note**: Newer provider, may have better availability

---

## Tier 3: SkyPilot Multi-Cloud (Automated Cost Optimization)

SkyPilot automatically finds the cheapest available GPUs across 12+ providers:

```bash
# Find cheapest 8x H100 across all clouds
sky show-gpus --all | grep H100

# Launch InferScope benchmark on cheapest available H100 cluster
sky launch inferscope-kimi-k25.yaml --use-spot
```

### SkyPilot-Compatible Providers
AWS, GCP, Azure, Lambda, RunPod, Kubernetes, OCI, Paperspace, Fluidstack, Coreweave, IBM, SCP

### Spot Instance Strategy
- Use `--use-spot` for 60-80% cost savings
- InferScope benchmarks are stateless and resumable — perfect for spot
- SkyPilot handles automatic preemption recovery

---

## Tier 4: Free Platforms (Limited GPU, Good for Development)

| Platform | GPU | Free Hours | Best For |
|----------|-----|-----------|----------|
| Google Colab | T4 (16GB) | 15-30/week | Development, testing small models |
| Kaggle | P100 (16GB) | 30/week | Dataset preparation, analysis |
| Lightning AI | T4/A10G | Monthly allocation | PyTorch experiments |
| Paperspace Gradient | Free tier | Limited | ML notebooks |

**Not suitable for Kimi-K2.5 or GLM-5** (insufficient VRAM), but useful for:
- Testing InferScope CLI and MCP against mock endpoints
- Preparing datasets and workload configurations
- Running support assessment and strategy planning

---

## Recommended Benchmark Strategy

### Phase 1: Validate on A100 (free/cheap)
- Use Google Cloud $300 credits or Lambda A100 ($1.10/hr)
- Run with smaller models (GLM-4-9B, Qwen2.5-32B) to validate pipeline
- Total cost: ~$10-30

### Phase 2: Kimi-K2.5 on H100 (spot instances)
- Use SkyPilot + RunPod/Lambda spot for 8x H100
- Run FP8 serving: `vllm serve moonshotai/Kimi-K2.5 -tp 8 --kv-cache-dtype fp8`
- InferScope experiment: `vllm-disagg-prefill-lmcache-kimi-k2`
- Estimated cost: $20-40/hr × 2-4 hours = **$40-160 total**

### Phase 3: GLM-5 on H100/H200 (spot instances)
- Run FP8: `vllm serve zai-org/GLM-5-FP8 -tp 8`
- InferScope experiment: `vllm-disagg-prefill-lmcache-glm5` or `dynamo-disagg-prefill-nixl-glm5`
- Estimated cost: $20-40/hr × 2-4 hours = **$40-160 total**

### Phase 4: NVFP4 on Blackwell (when available)
- Requires B200/B300 GPUs (Blackwell architecture)
- 4x B200 sufficient for both models in NVFP4
- Watch for RunPod/Lambda B200 spot availability

---

## Partnership Opportunities for Free Compute

### NVIDIA Inception Program
- Free DGX Cloud credits for AI startups
- Access to NGC containers and optimized models
- Technical mentorship
- **URL**: https://www.nvidia.com/en-us/startups/

### AMD MI300X Developer Access
- ROCm ecosystem development access
- Potential benchmark collaboration (AMD published Kimi-K2.5 MXFP4 optimizations)
- **Contact**: ROCm developer relations

### SkyPilot (UC Berkeley)
- Active InferScope collaboration partner
- Potential shared benchmark infrastructure
- Multi-cloud GPU orchestration for reproducible results
