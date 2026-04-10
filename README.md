# IARS Nanoswimmer Simulator

## Overview

This project implements an advanced simulation of nanoswimmer-based tumor targeting using the Individual Adaptive Regulation Strategy (IARS). The system models how nanoscale agents collectively sense biological gradients, evaluate reliability, and dynamically navigate toward target regions under realistic constraints.

Unlike earlier static models, this updated framework integrates:
- Dynamic motion
- Adaptive behavioral modes (forage vs chase)
- Vessel-constrained navigation
- Multi-gradient testing environments

The simulation demonstrates how decentralized swarm intelligence can achieve stable, accurate, and scalable navigation in noisy environments.

---

## Project Files Structure

```

├── nanoswimmer_simulation.py        # Core simulation logic (main code)
├── IARS-Nanoswimmer-Simulator.ipynb # Jupyter notebook (experiments & visualization)

├── bgf_exports_smoke/               # Output data & generated results
│   ├── bgf_metric_comparison.png
│   ├── bgf_movement_patterns.png
│   └── bgf_summary.csv

├── research-paper.pdf              # Final research paper
├── final-report.pdf                # Project report (university submission)


```

---

## Key Features

- Multiple Biological Gradient Fields:
  - Sphere
  - Matyas
  - Ackley
  - Easom

- Local Gradient Estimation using finite differences
- Fitness-based reliability scoring
- Elite selection mechanism
- Weighted vector fusion for global direction
- Adaptive behavior switching:
  - Forage (exploration)
  - Chase (target-driven movement)

- Vessel-constrained motion (biologically inspired pathways)
- Real-time trajectory tracking
- Detection statistics and convergence monitoring

---

## Core Algorithm

The system follows this pipeline:

```

Initialize Agents
→ Sense Local Gradient
→ Compute Fitness
→ Select Elite Agents
→ Fuse Directions
→ Choose Behavior (Forage / Chase)
→ Move Agents
→ Detect Target
→ Repeat

```

Global direction is computed using:

```

V = Σ (f_i * v_i)

````

Where:
- f_i = fitness of agent
- v_i = direction vector

---

## How to Run

### Option 1: Python Script

```bash
python nanoswimmer_simulation.py
````

---

### Option 2: Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
IARS-Nanoswimmer-Simulator.ipynb
```

---

## Results

The simulation demonstrates:

* High detection rates (up to ~95%+)
* Stable convergence toward target
* Reduced noise through elite filtering
* Clear trajectory formation
* Effective navigation under constraints

Output files are stored in:

```
bgf_exports_smoke/
```

---

## Outputs Included

* Movement trajectories visualization
* Gradient field comparisons
* Detection performance metrics
* CSV summary of results

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* Jupyter Notebook

---

## Strengths

* Robust to noise and uncertainty
* Fully decentralized system
* Scalable to large agent populations
* Biologically inspired design
* Realistic motion with constraints

---

## Limitations

* Simplified fluid dynamics
* Idealized gradient models
* No inter-agent communication
* Behavior switching is rule-based (not learned)

---

## Applications

* Targeted drug delivery
* Tumor localization
* Precision medicine
* Molecular communication systems

---

## Future Work

* Reinforcement learning for adaptive behavior
* Real biological data integration
* Advanced fluid dynamics modeling
* Inter-agent communication systems
* Hardware-level nanoswimmer implementation

---

## License
```
This project is licensed under the MIT License.
```
---

## Disclaimer
```
This project is intended for academic and research purposes only. It is not a medical or clinical system.

```


