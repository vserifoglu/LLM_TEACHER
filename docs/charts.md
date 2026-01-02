# Charts and Visualizations

This document contains various charts and visualizations for the LLM fine-tuning workflow.

---

## Overview: Workflow Pipeline
```
graph TD
    %% Styles
    classDef dataset fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,rx:10,ry:10;
    classDef model fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:5,ry:5;
    classDef reward fill:#ffebee,stroke:#c62828,stroke-width:2px;
    classDef output fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    %% --- DATA LAYER ---
    subgraph "Dataset (30k samples)"
        Math["Math<br/>GSM8K"]:::dataset
        Coding["Coding<br/>MBPP"]:::dataset
        Science["Science<br/>SciQ, ARC"]:::dataset
        Logic["Logic<br/>StrategyQA"]:::dataset
        Creative["Creative Writing<br/>WritingPrompts"]:::dataset
        Ideation["Creative Ideation<br/>Dolly"]:::dataset
        Summary["Summarization<br/>XSum"]:::dataset
    end

    %% DISCO Sampling
    Math --> DISCO[DISCO Sampling<br/>Temperature=0.5]:::process
    Coding --> DISCO
    Science --> DISCO
    Logic --> DISCO
    Creative --> DISCO
    Ideation --> DISCO
    Summary --> DISCO

    %% --- MODEL LAYER ---
    subgraph "Model Architecture"
        Gemma["Gemma 3 1B IT<br/>Base Model"]:::model
        LoRA["LoRA Adapter<br/>Rank=16, Alpha=32"]:::process
        Gemma --> LoRA
    end

    %% --- TRAINING LAYER ---
    subgraph "GRPO Training Loop"
        DISCO --> Prompts["Formatted Prompts<br/>+ System Instruction"]:::process
        Prompts --> Generate["Generate 4 Responses<br/>per Prompt"]:::process
        LoRA --> Generate
        
        Generate --> Rewards{{"Reward Calculation"}}:::reward
        
        subgraph "Reward Functions"
            Format["XML Format Check<br/>(Hard Gate)"]:::reward
            Verifiable["Correctness Check<br/>Math/Coding/Science"]:::reward
            Creative_R["Heuristic Quality<br/>Length/Diversity"]:::reward
        end
        
        Rewards --> Format
        Rewards --> Verifiable
        Rewards --> Creative_R
        
        Rewards --> Policy["Policy Update<br/>KL Penalty β=0.04"]:::process
        Policy --> LoRA
    end

    %% --- OUTPUT ---
    LoRA --> Final["Final Model<br/>LoRA Checkpoint"]:::output
    Final --> Inference["Inference<br/>Greedy Decoding"]:::output

    %% Clean layout
    linkStyle default stroke:#666,stroke-width:1.5px

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