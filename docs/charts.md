# Charts and Visualizations

This document contains various charts and visualizations for the LLM fine-tuning workflow.

---

## Overview: Workflow Pipeline
```
graph TD
    %% Define Styles
    classDef dataset fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,rx:10,ry:10;
    classDef model fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:5,ry:5;
    classDef storage fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,stroke-dasharray: 5 5;
    classDef note fill:#ffffff,stroke:#666666,stroke-width:1px,stroke-dasharray: 3 3;
    classDef reward fill:#ffebee,stroke:#c62828,stroke-width:2px;

    %% --- PHASE 1: The Baseline ---
    subgraph "Session 1: The Baseline (SFT)"
        Gemma[Google Gemma 2B Base]:::model -->|Load| SFT_Train[Train: Supervised Fine-Tuning]:::process
        MathData1["Math Dataset - GSM8K"]:::dataset --> SFT_Train
        SFT_Train -->|Save Weights| V1_Ckpt["Variant 1 Checkpoint"]:::storage
    end

    %% --- PHASE 2: The Specialist ---
    subgraph "Session 2: The Specialist (RL Math)"
        V1_Ckpt -->|Load Weights| GRPO_Math[Train: GRPO Reinforcement Learning]:::process
        MathData2["Math Dataset"]:::dataset --> GRPO_Math
        Reward_Math{{"Reward: Correctness"}}:::reward --> GRPO_Math
        GRPO_Math -->|Save Weights| V2_Ckpt["Variant 2 Checkpoint"]:::storage
    end

    %% --- PHASE 3: The Generalist ---
    subgraph "Session 3: The Generalist (RL Transfer)"
        V2_Ckpt -->|Load Weights| GRPO_Mix[Train: GRPO Mixed Domain]:::process
        CreativeData["Creative & Logic Data"]:::dataset --> GRPO_Mix
        MathData3["Math Data - 20% Mix"]:::dataset --> GRPO_Mix
        Reward_Rubric{{"Reward: Rubrics + Format"}}:::reward --> GRPO_Mix
        GRPO_Mix -->|Final Output| V3_Final["Variant 3 - Final Model"]:::model
    end

    %% --- Parallel Branching Note ---
    Note_Branching["Optimization Tip: <br/>Run Session 2 multiple times in parallel<br/>with different Learning Rates to find<br/>the best V2 before starting Session 3"]:::note
    V1_Ckpt -.-> Note_Branching
    Note_Branching -.-> GRPO_Math

    %% Connect the flow visually
    linkStyle 0 stroke:#4CAF50,stroke-width:2px;
    linkStyle 1 stroke:#2196F3,stroke-width:2px;
    linkStyle 2 stroke:#9C27B0,stroke-width:2px;
    linkStyle 3 stroke:#4CAF50,stroke-width:2px;
    linkStyle 4 stroke:#2196F3,stroke-width:2px;
    linkStyle 5 stroke:#FF9800,stroke-width:2px;
    linkStyle 6 stroke:#4CAF50,stroke-width:2px;
    linkStyle 7 stroke:#2196F3,stroke-width:2px;
    linkStyle 8 stroke:#9C27B0,stroke-width:2px;
    linkStyle 9 stroke:#FF9800,stroke-width:2px;
    linkStyle 10 stroke:#9C27B0,stroke-width:2px;
    linkStyle 11 stroke:#666666,stroke-width:1px,stroke-dasharray: 3 3;
    linkStyle 12 stroke:#666666,stroke-width:1px,stroke-dasharray: 3 3;

```

---

## Jax Echosystem Architecture
```
graph TD
  %% Root
  JAX["JAX<br/>(arrays, jit/vmap/grad)"]

  %% Direct-on-JAX utilities
  ml_dtypes["ml_dtypes<br/>(extra ML dtypes)"]
  chex["chex<br/>(testing/utilities)"]

  %% Core NN & training libraries
  Flax["Flax (NNX)<br/>(neural network modules)"]
  Optax["Optax<br/>(optimizers & gradient transforms)"]
  Orbax["Orbax<br/>(checkpointing)"]
  Grain["Grain / tf.data<br/>(data input pipelines)"]

  %% High-level LLM systems
  MaxText["MaxText<br/>(LLM training & inference harness)"]
  Tunix["Tunix<br/>(LLM post-training & alignment)"]

  %% Edges: who depends on whom
  JAX --> ml_dtypes
  JAX --> chex
  JAX --> Flax
  JAX --> Optax
  JAX --> Orbax
  JAX --> Grain

  Flax --> MaxText
  Optax --> MaxText
  Orbax --> MaxText
  Grain --> MaxText

  %% Tunix sits on top of Flax/Optax/JAX, often used with MaxText
  Flax --> Tunix
  Optax --> Tunix
  MaxText -. optional orchestration .- Tunix

```