# Epistemic Pipeline Analysis - Complete Documentation

## 📚 Document Index

This folder contains comprehensive analysis and simulation materials for the **Epistemic Pipeline** - the 4-layer self-learning knowledge system in AGI.

### Starting Points

**New to the epistemic pipeline?** Start here in order:

1. **EPISTEMIC_PIPELINE_SUMMARY.md** ← **START HERE** (5 min read)
   - High-level overview of what the pipeline is
   - Quick explanation of the 4 layers
   - Real-world example walkthrough
   - Integration points and key insights

2. **EPISTEMIC_PIPELINE_DIAGRAMS.md** (10 min read)
   - 9 visual ASCII diagrams
   - Theory lifecycle state machine
   - Complete data flow for single cycle
   - Confidence mechanics visualized
   - Algorithm flows (focus selection, curiosity, decay)
   - 3-cycle end-to-end example

3. **EPISTEMIC_PIPELINE_ANALYSIS.md** (30 min read)
   - Deep dive into each layer
   - Data structure definitions
   - Method documentation
   - Bootstrap axioms explained
   - Multi-cycle simulation (Cycles 1-20)
   - Edge cases and design decisions

4. **EPISTEMIC_PIPELINE_CODE_REFERENCE.md** (20 min read)
   - Actual method signatures
   - Pseudo-code for all layers
   - Complete cycle example with real values
   - Neo4j schema constraints
   - Async lock coordination details
   - Confidence score interpretation

### Executable Materials

5. **epistemic_simulation.py** (runnable)
   - Python simulation (no Neo4j required)
   - Deterministic/mocked responses
   - Demonstrates all 4 layers
   - Shows actual cycle execution
   - Run with: `python3 epistemic_simulation.py`

---

## 🎯 Quick Answers

### "What is the epistemic pipeline?"
A 4-layer feedback system that enables AGI to build reliable world models through:
1. **World Model**: Store theories with confidence scores
2. **Explorer**: Execute grounded tests
3. **Analyzer**: Compare predictions vs reality
4. **Optimizer**: Update confidence based on results

→ See **EPISTEMIC_PIPELINE_SUMMARY.md**

### "How does it learn?"
```
Theory (confidence=0.1) 
  → Test it via COGNITION mode
  → Compare prediction vs observation
  → If match: confidence ↑ (e.g., 0.1 → 0.43)
  → If mismatch: confidence ↓
  → If confidence ≤ 0.0: delete
  → If new hypothesis found: inject as new theory
```

→ See **EPISTEMIC_PIPELINE_ANALYSIS.md** § "Multi-Cycle Example"

### "How does it avoid stale beliefs?"
Temporal decay formula (120+ day old theories pulled toward neutral 0.5) forces re-validation.

→ See **EPISTEMIC_PIPELINE_DIAGRAMS.md** § "Temporal Decay Formula"

### "Can I see it in action?"
Yes! Run the simulation:
```bash
python3 epistemic_simulation.py
```

→ See **epistemic_simulation.py**

### "How does curiosity work?"
Every 5 cycles, instead of testing a theory, the system explores a curiosity topic (e.g., "advanced physics theories"). Topics with fewer observations are prioritized.

→ See **EPISTEMIC_PIPELINE_DIAGRAMS.md** § "Curiosity Topic Selection"

### "What happens when a theory fails?"
```
Confidence decreases by: suggested_adjustment × confidence_in_data

Examples:
- Strong contradiction, high quality obs: -0.35 × 0.95 = -0.33 (big drop)
- Weak contradiction, noisy obs: -0.2 × 0.3 = -0.06 (small drop)

Eventually: confidence ≤ 0.0 → theory deleted
```

→ See **EPISTEMIC_PIPELINE_DIAGRAMS.md** § "Theory Lifecycle"

### "How are new theories created?"
The Analyzer can suggest `new_hypothesis` from observations. If non-null, Optimizer creates new Theory with:
- text = new_hypothesis
- confidence = 0.1 (fresh start)
- attempts = 0 (untested)

→ See **EPISTEMIC_PIPELINE_ANALYSIS.md** § "Layer 4: Optimizer"

---

## 📊 Architecture Overview

```
                    EPISTEMIC LOOP
                          │
                          ▼
            ┌─────────────────────────┐
    SELECT  │   LAYER 1: WORLD MODEL  │
    ─────→  │   (Neo4j Graph)         │
            │   • Theories            │
            │   • Confidence scores   │
            │   • History             │
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
   EXECUTE  │   LAYER 2: EXPLORER     │
    ─────→  │   (Grounded Actions)    │
            │   • COGNITION mode      │
            │   • Tools + sensors     │
            │   • Returns observation │
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
   ANALYZE  │   LAYER 3: ANALYZER     │
    ─────→  │   (Error Estimation)    │
            │   • Compare prediction  │
            │   • vs observation      │
            │   • Returns error sig   │
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
   UPDATE   │   LAYER 4: OPTIMIZER    │
    ─────→  │   (Feedback Control)    │
            │   • Update confidence   │
            │   • Inject hypotheses   │
            │   • Delete weak theories│
            └────────────┬────────────┘
                         │
                    (next cycle)
```

---

## 🔑 Key Concepts

### Theory Lifecycle

```
NEW (0.1) → TESTING → STRONG (>0.5) ✓
              ↓
            WEAK (<0.3) ✗ → DELETE
              ↓
            STALE (120+ days) → decay toward 0.5
```

### Confidence Adjustment

```
adjustment = suggested_adjustment × confidence_in_data

Intuition: 
- High quality evidence: full adjustment applied
- Low quality evidence: adjustment dampened
- Prevents noisy observations from shifting belief too much
```

### Temporal Decay (Anti-Staleness)

```
recency_weight = max(0.5, 1.0 - (age_days / 120))

0 days   → weight=1.0 (fresh, no decay)
60 days  → weight=0.5 (aging, pulled toward neutral)
120+ days → weight=0.5 (stale, forced re-validation)
```

### Focus Selection

```python
if cycles_since_curiosity >= 5:
    explore_curiosity()  # ~20% of cycles
else:
    test_theory()        # ~80% of cycles
```

### Async Parallelism

```
SELECT [lock]
  ↓
EXPLORE & ANALYZE [no lock - can parallelize]
  ↓
OPTIMIZE [lock - exclusive write]
```

---

## 📖 Reading Guide by Use Case

### "I want a quick overview"
→ **EPISTEMIC_PIPELINE_SUMMARY.md** (5 min)

### "I need to understand the theory lifecycle"
→ **EPISTEMIC_PIPELINE_DIAGRAMS.md** § "Theory Lifecycle" (5 min)

### "I need to know the exact formulas"
→ **EPISTEMIC_PIPELINE_DIAGRAMS.md** § "Temporal Decay" (5 min)

### "I want to see a complete cycle with numbers"
→ **EPISTEMIC_PIPELINE_ANALYSIS.md** § "Simulation" (15 min)

### "I need method signatures and pseudo-code"
→ **EPISTEMIC_PIPELINE_CODE_REFERENCE.md** (20 min)

### "I want to run a working simulation"
→ Run **epistemic_simulation.py** (2 min)

### "I need deep implementation understanding"
→ **EPISTEMIC_PIPELINE_ANALYSIS.md** + **EPISTEMIC_PIPELINE_CODE_REFERENCE.md** (60 min)

### "I want visuals and diagrams"
→ **EPISTEMIC_PIPELINE_DIAGRAMS.md** (30 min)

---

## 📌 Repository Memory

Additional context stored in `/memories/repo/`:

- **epistemic_pipeline_architecture.md**: Quick reference summary
- **vision_architecture_overview.md**: Related vision system
- **sense_initialization_architecture.md**: Related sense system
- **testing_pipelines_guide.md**: Testing methodology

---

## 🔗 Code References

The actual implementation code is in:
```
src/epistemic/
├── layer1_world_model.py    # Neo4j knowledge graph
├── layer2_exploration.py    # COGNITION execution
├── layer3_analyzer.py       # Error estimation
├── layer4_optimizer.py      # Feedback control
├── main_loop.py             # Orchestration
├── types.py                 # Data structures
└── __init__.py
```

---

## ⚙️ Configuration

Default parameters (from `main_loop.py`):

```python
curiosity_topics = [
    "advanced physics theories",
    "artificial intelligence architecture", 
    "local software environment",
    "physical surroundings"
]

curiosity_interval = 5           # Explore curiosity every 5 theory tests
max_theory_attempts = 3          # Don't test same theory >3 times
pause_seconds = 2.0              # Delay between cycles
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "epistemic123"
```

---

## 🚀 Getting Started (3-Step Path)

### Step 1: Understand the System (5 min)
Read: **EPISTEMIC_PIPELINE_SUMMARY.md**

### Step 2: Visualize the Architecture (15 min)
Read: **EPISTEMIC_PIPELINE_DIAGRAMS.md**

### Step 3: See It Work (5 min)
Run: `python3 epistemic_simulation.py`

---

## 📝 Notes

- All times are estimates and depend on reading speed
- The simulation doesn't require Neo4j or LLM API calls
- Diagrams use ASCII art for universal compatibility
- Code reference includes both signatures and pseudo-code
- Deep dives available for specific topics

---

## ✅ What You Now Understand

After reading these materials, you'll know:

✓ How the epistemic pipeline learns from observations  
✓ What theories are and how confidence scores work  
✓ How temporal decay prevents stale beliefs  
✓ Why the system balances theory testing with curiosity  
✓ How hypotheses are generated and injected  
✓ What happens during a complete cycle  
✓ How async parallelism works  
✓ Why confidence adjustments are weighted by observation quality  
✓ How the system recovers from failures  
✓ How the world model evolves over time  

---

## 🎓 Advanced Topics

For deeper understanding, see relevant sections:

- **Temporal Decay**: EPISTEMIC_PIPELINE_DIAGRAMS.md § "Temporal Decay Formula"
- **Async Coordination**: EPISTEMIC_PIPELINE_DIAGRAMS.md § "Write Lock Coordination"
- **Hypothesis Injection**: EPISTEMIC_PIPELINE_DIAGRAMS.md § "Hypothesis Injection Workflow"
- **Focus Selection**: EPISTEMIC_PIPELINE_DIAGRAMS.md § "Curiosity Topic Selection Algorithm"
- **Confidence Mechanics**: EPISTEMIC_PIPELINE_DIAGRAMS.md § "Confidence Adjustment Mechanics"

---

**Last Updated**: May 8, 2026  
**Scope**: Complete analysis of src/epistemic/ module  
**Documents**: 5 (4 markdown + 1 Python simulation)  
**Total Content**: ~40,000 words + 9 diagrams + runnable code
