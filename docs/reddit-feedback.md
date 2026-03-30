# kernel-anvil Reddit Feedback Report

**Generated:** 2026-03-28
**Threads monitored:**
- r/LocalLLaMA: 76 upvotes, 96% ratio, 31 comments
- r/ROCm: 13 upvotes, 93% ratio, 11 comments

---

## 1. Bug Reports / Issues

### [ACTIONABLE] Repo clone / link broken
- **u/spaceman3000** (score 1, ~1h ago, r/LocalLLaMA): "Can't clone your repo for some reason. It's asking for username and password and doesn't accept mine. **EDIT: repo link you're providing in manual on github doesn't exist.** EDIT2: Doesn't work on Strix Halo: `No GPU available. gguf-optimize requires a CUDA/ROCm GPU.`"
- This user hit TWO issues: a broken repo link in the docs AND Strix Halo GPU detection failure.

### [ACTIONABLE] Strix Halo GPU not detected
- **u/spaceman3000** (score 1, ~9m ago, r/LocalLLaMA): "Tried it on strix halo. Doesn't work, doesn't see the GPU."
- **u/Own_Suspect5343** (score 1, ~1h ago, r/LocalLLaMA): "I test it 1 hour ago. I don't get improvement because memory-bound limit("
- Multiple users confirming Strix Halo does not work despite OP claiming RDNA3.5/4 support.

### [ACTIONABLE] Triton install fails
- **u/uber-linny** (score 1, ~10m ago, r/LocalLLaMA): "im going crazy, im getting this error: `ERROR: Could not find a version that satisfies the requirement triton>=3.0 (from kernel-anvil) (from versions: none)` `ERROR: No matching distribution found for triton>=3.0`"
- Likely a pip/platform issue -- triton wheels may not be available for their Python version or OS.

### [ACTIONABLE] Missing llama.cpp patch branch
- **u/fallingdowndizzyvr** (score 5, ~3h ago, r/LocalLLaMA): "I can't find that branch. Link please."
- **u/dulouie2** (score 5, ~2h ago, r/LocalLLaMA): "I can't find a llama.cpp fork on your github profile?"
- **u/spaceman_** (score 2, ~46m ago, r/LocalLLaMA): "Same. I'd love to do some testing (I have a pretty wide array of AMD hardware), but can't."
- **u/OfficialXstasy** (score 1, ~2h ago, r/LocalLLaMA): "The llama.cpp patch (~50 lines to mmvq.cu) is on branch `smithy-shape-configs`? What repo? Can you link to this branch?"
- **u/woct0rdho** (score 1, ~9m ago, r/LocalLLaMA): "Also waiting for more information on the branch smithy-shape-configs."
- **5 people** cannot find the smithy-shape-configs branch. This is the single most repeated issue.

---

## 2. Feature Requests

### RDNA4 support
- **u/RayIsLazy** (score 3, ~4h ago, r/ROCm): "What about RDNA4?"
- OP replied that RDNA3.5 & 4 are supported.

### RDNA / RDNA2 support
- **u/Dryw_Filtiarn** (score 1, ~3h ago, r/ROCm): "Would support for RDNA and RDNA2 also be possible?"
- **u/Sisuuu** (score 5, ~3h ago, r/LocalLLaMA): "Amazing work! Are you possibly working or going to implement a solution for RDNA2(RX6800) also?"

### Vulkan backend support
- **u/SlaveZelda** (score 3, ~2h ago, r/LocalLLaMA): "No vulkan?"

### CUDA support
- **u/JayPSec** (score 2, ~2h ago, r/LocalLLaMA): "Could this possible be applied to cuda as well?"

### Intel Arc / Qualcomm Adreno support
- **u/SkyFeistyLlama8** (score 1, ~2h ago, r/LocalLLaMA): "Can this be done for different GPU architectures like on Intel Arc and Qualcomm Adreno? On a slightly related note, maybe Adreno is doable because it's a distant relative of Radeon."

### iGPU support
- **u/Final-Frosting7742** (score 1, ~3h ago, r/LocalLLaMA): "Does it accelerate iGPU too?"

### Strix Halo / APU support (confirmed needed despite OP claim)
- **u/Own_Suspect5343** (score 13, ~4h ago, r/LocalLLaMA): "Does it work with rdna 3.5 (strix halo)?"
- **u/-deleled-** (score 1, ~3h ago, r/ROCm): "Specific to RDNA3? What about RDNA3.5 on Strix Halo for example?"
- **u/Professional_Quit_31** (score 2, ~3h ago, r/LocalLLaMA): "Strix Support would be nice"

### Multi-GPU with different architectures
- **u/spaceman_** (score 1, ~50m ago, r/LocalLLaMA): "I run some models on mixed GPUs (RDNA3 and RDNA4 in the same system), is there a way to use different optimizations on both? Would one set of optimizations bring any benefits?"

### Batching / throughput optimization
- **u/shing3232** (score 0, ~2h ago, r/LocalLLaMA): "Do you need to have tuning for big batching? cause llama cpp is well-known for low throughput compare to VLLM"

### Curated inference improvements tracker
- **u/t4a8945** (score 15, ~3h ago, r/LocalLLaMA): "It's becoming hard for me to track the latest improvements for inference. Does anyone know a place that curates this? Something like 'select your hardware, get latest improvements news'?"

---

## 3. Questions

### Why is the baseline so slow?
- **u/superdariom** (score 1, ~4h ago, r/ROCm): "I'm running the same GPU and same qwen 3.5 but on the standard llama I'm already getting 20 t/s very happy to try this out and get a speed up but I'm not sure why yours is at 12"
  - **OP reply:** "Has to do with turboquant. This is with both turboquant and the kernel optimizer."
- **u/flavio_geo** (score 1, ~1h ago, r/ROCm): "I run on xtx 7900, without this I already get about 30 t/s on llama rocm and 40 t/s on llama vulkan. I dont understand why your values are low like this. Also PP is better on vulkan in this model for me, specially if you bench longer context. I honestly just use llama boilerplate config. Using the qwen3.5 27b ud_q4_xl from Unsloth. I run on Ubuntu 24.04."
- **u/noctrex** (score 5, ~1h ago, r/LocalLLaMA): "I'm sorry, maybe something with your installation is not working correctly? On what system are you running it? That's awfully slow for a 7900XTX. I'm on Windows 11, and using the latest mainline llama.cpp, I get about 40 tps with Qwen3.5-27B (UD-Q4_K_XL) with the Vulkan backend on my 7900XTX."
  - Posted benchmark screenshots showing ~40 tps baseline.

**This is a recurring theme -- multiple people question the 12 tok/s baseline as unusually low for a 7900 XTX. The turboquant explanation needs to be more prominent in the post/README.**

### How does mmvq.cu apply to AMD?
- **u/HopePupal** (score 4, ~3h ago, r/LocalLLaMA): "yeah so i'm really excited to hear how patching `mmvq.cu` works on an AMD card"

### Do I need to recompile llama.cpp?
- **u/-deleled-** (score 1, ~1h ago, r/ROCm): "Should I recompile my llama.cpp and link it to kernel-anvil? I read the readme and didn't see anything mentioned about the llama.cpp compilation specifically"

### What exactly is being optimized?
- **u/spaceman_** (score 1, ~50m ago, r/LocalLLaMA): "What are you optimizing, exactly? The shape of the work groups?"
  - **u/OfficialXstasy** provided an excellent detailed explanation of per-shape nwarps/rows_per_block tuning (score 1).

---

## 4. Positive Feedback

- **u/t4a8945** (score 15, r/LocalLLaMA): "Amazing. I'm impressed by anything like this done by individuals. Well done."
- **u/madsheepPL** (score 2, r/LocalLLaMA): "Good job. Love the name :)"
- **u/Sisuuu** (score 5, r/LocalLLaMA): "Amazing work!"
- **u/AwayLuck7875** (score 1, r/ROCm): "12-20 token small data server normal" (validating the baseline numbers)

---

## 5. Technical Corrections

### [ACTIONABLE] Triton kernel may not reproduce HIP kernel performance
- **u/woct0rdho** (score 1, ~9m ago, r/LocalLLaMA): "I had a quick read of your repo and I'd say the Triton kernel may not accurately reproduce the performance of the HIP kernel in llama.cpp. It would be better to actually compile the HIP kernel with various parameters then benchmark them."
- This is a legitimate technical concern about profiling methodology.

### Baseline numbers questioned
- Multiple users (noctrex, flavio_geo, superdariom) report 20-40 tok/s baseline on the same GPU without any optimizations, questioning the 12 tok/s starting point. The turboquant context needs clarification -- the 2.25x claim may be misleading if the baseline is artificially low due to turboquant overhead.

### Skepticism about verifiability
- **u/OfficialXstasy** (score 7, ~2h ago, r/LocalLLaMA): "Be a little careful with new projects missing bits and pieces. Not saying this isn't legit, but there's omitted information from the first post which makes these claims unverifiable. Let's wait for a much needed reply from the author before we send this post flying. I'm also on gfx1100, so I do hope it's true, it's just not 100% verifiable yet."

---

## 6. Spam / Low Quality

- **u/Ok-Drawing-2724** (score 0, ~1h ago, r/LocalLLaMA): "Handy tool for AMD users. ClawSecure is useful when trying new optimization tools.... just to make sure nothing weird happens with the model or GPU." -- Appears to be a soft plug for an unrelated product.

---

## Priority Actions

| Priority | Issue | Impact |
|----------|-------|--------|
| **P0** | Publish the `smithy-shape-configs` llama.cpp branch (5 people asking) | Blocks all external testing |
| **P0** | Fix broken repo link mentioned in docs/README | Blocks clone |
| **P1** | Fix Strix Halo GPU detection (multiple confirmed failures) | Breaks claimed RDNA3.5/4 support |
| **P1** | Clarify baseline numbers -- 12 tok/s vs community's 20-40 tok/s | Credibility issue |
| **P1** | Address triton install failure (triton>=3.0 not found) | Blocks install |
| **P2** | Clarify whether llama.cpp recompilation is needed | README gap |
| **P2** | Address woct0rdho's point about Triton vs HIP kernel accuracy | Technical credibility |
| **P3** | Consider RDNA2, Vulkan, CUDA support requests | Feature planning |
