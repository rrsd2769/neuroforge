# NeuroForge — Demo Script

Two versions: a 60-second verbal overview and a 5-minute technical walkthrough.

---

## 60-Second Version (phone screen / recruiter)

> "NeuroForge is a machine learning experimentation engine I built from scratch over 11 
> days. The interesting part isn't the ML — it's the architecture. The entire system 
> uses hexagonal architecture, so PyTorch, the REST API, and the dashboard are all 
> pluggable infrastructure around a pure Python domain layer that has zero framework 
> dependencies.
>
> In practice it means: you can swap the trainer from PyTorch to JAX without changing a 
> single line of domain code. You can replace the file-based experiment tracker with a 
> database adapter. The API and dashboard just talk to ports.
>
> It has a Streamlit dashboard where you can build architectures layer-by-layer, train 
> them on CIFAR-10, and compare results across runs. I seeded it with 6 experiments 
> across different depths, optimizers, and regularization strategies — dropout 
> outperformed everything else at 51% top-1, which is meaningful given the tiny data 
> subsets used."

---

## 5-Minute Walkthrough (technical interview / portfolio review)

### Stop 1 — Home page (30 seconds)
- Point to the 🟢 API status — API and dashboard are separate services
- Point to the best model banner — Conv-Dropout at 51.2%
- Point to the accuracy trend chart — shows the progression across experiments
- **Say:** "The trend line shows dropout regularization as the inflection point — 
  Conv-Medium plateaued, Conv-Dropout jumped 4 points"

### Stop 2 — Run Experiment page (60 seconds)
- Show the layer builder — add a conv layer live
- Point to the validation warnings (no flatten = warning, no dense at end = warning)
- Point to the estimated time caption
- **Say:** "The architecture builder constructs a JSON payload that hits 
  POST /experiments/run. The API creates the experiment record, runs training, 
  and returns results. The dashboard has no PyTorch dependency — it's pure HTTP."

### Stop 3 — Experiments page (60 seconds)
- Select Conv-Dropout and SGD-Momentum in the comparison multiselect
- Show the head-to-head grouped bar chart render
- Point to the metrics table below it
- **Say:** "Same architecture, same data, different optimizer. Adam beats SGD by 
  13 points here — that's a real finding not a synthetic demo."

### Stop 4 — Results page (30 seconds)
- Paste Conv-Dropout's experiment ID
- Show the 4 metric cards, training config, layer stack, pie chart
- **Say:** "Every experiment is fully reproducible — config, architecture, 
  and results all persisted."

### Stop 5 — Codebase (60 seconds, share screen on IDE)
- Open `neuroforge_core/domain/interfaces/` — show the Trainer port
- Open `neuroforge_core/infrastructure/pytorch_trainer.py` — show it implements the port
- Open `neuroforge_core/application/train_model_use_case.py` — show it only knows the port
- **Say:** "The use case takes a Trainer interface. It doesn't know if it's PyTorch, 
  a mock, or anything else. That's what makes the 51 tests fast — the domain tests 
  never touch real training."

---

## Questions You'll Get

**"Why hexagonal architecture for an ML project?"**
> Most ML code is notebooks or scripts — untestable, unreplaceable, coupled to specific 
> frameworks. I wanted to demonstrate that ML systems can be held to the same engineering 
> standards as any backend service. The architecture patterns didn't slow the build — 
> they sped up testing and let me swap the API layer on Day 8 without touching domain code.

**"What would you do differently?"**
> The experiment tracker uses flat JSON files — a SQLite adapter would be a straightforward 
> swap given the port interface. I'd also add a proper async training queue so the API 
> isn't blocked during training. Both are one-adapter changes, which is the point of 
> the architecture.

**"What accuracy could you get with full training?"**
> With all 50k CIFAR-10 training samples and 50 epochs, a VGG-style conv net hits 
> 85-90%. The demo uses 1-3k samples for speed. The architecture that won here 
> (dropout + wide convs) is consistent with what works at full scale.