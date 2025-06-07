---
layout: default
---
```mermaid
graph TD
    %% Input Processing
    A["🔤 Input Prompt<br/>What is the capital of France?<br/><i>Raw text from user</i>"] --> B["📝 Tokenization<br/>Split text into tokens<br/>What | is | the | capital | of | France | ?"]
    B --> C["🔢 Token IDs<br/>[1234, 567, 89, 1011, 1213, 445, 63]<br/><i>Convert tokens to numbers</i>"] --> D["📊 Token Embedding<br/>Map each ID to dense vector<br/><i>e.g., 1234 → [0.2, -0.1, 0.8, ...]</i>"]
    D --> E["📍 + Positional Encoding<br/>GPT: Learned positional embeddings (not sinusoidal)<br/>Original Transformer: Sinusoidal patterns<br/><i>Modern LLMs learn position representations during training</i>"]

    %% Decoder Stack Processing
    E --> DecoderStack["🧠 N DECODER LAYERS (×12 to ×96+ in modern LLMs)<br/>Each layer progressively builds understanding:<br/>Early layers: Basic patterns, syntax<br/>Middle layers: Semantics, reasoning<br/>Late layers: Complex abstractions, predictions"]

    DecoderStack --> F6["🧠 Final Layer Output<br/>Rich contextual representation<br/><i>Ready for prediction</i>"]

    %% What happens in each layer
    subgraph Layer["🔍 DECODER LAYER - GPT-2/3 PRE-NORM ARCHITECTURE"]
        G0["🔧 Layer Norm<br/>Pre-normalization: norm(x) before attention<br/><i>Applied BEFORE sub-layer, not after</i>"] --> G1["🎯 MASKED MULTI-HEAD SELF-ATTENTION<br/>• Attention(Q,K,V) = softmax(QK^T/√d_k)V<br/>• √d_k scaling prevents vanishing gradients<br/>• 🚫 CAUSAL MASK: -∞ for future positions<br/>• Heads: 12 (GPT-2-small), 25 (GPT-2-medium), 32 (GPT-2-large), 96 (GPT-3-175B)<br/><i>Core context understanding mechanism</i>"]
        G1 --> G2["➕ Residual Add<br/>x + attention(norm(x))<br/><i>Skip connection preserves gradients</i>"]
        G2 --> G3["🔧 Layer Norm<br/>Pre-normalization: norm(x) before MLP<br/><i>Applied BEFORE MLP, not after</i>"]
        G3 --> G4["🔄 Feed Forward Network (MLP)<br/>Linear(d_model → 4×d_model) → ReLU → Linear(4×d_model → d_model)<br/>GPT-2: 768→3072→768, GPT-3: 12288→49152→12288<br/><i>Position-wise processing with 4× expansion rule</i>"]
        G4 --> G5["➕ Residual Add<br/>x + mlp(norm(x))<br/><i>Final layer output with skip connection</i>"]
    end

    %% Output Processing
    F6 --> H["🎯 Language Model Head<br/>Linear layer: hidden_size → vocab_size<br/><i>Projects to 50,000+ possible tokens</i>"]
    H --> I["📊 Softmax<br/>Convert logits to probabilities<br/><i>All probabilities sum to 1.0</i>"]
    I --> J["🎲 Probability Distribution<br/>P(Paris) = 0.73, P(London) = 0.02, P(Berlin) = 0.01<br/><i>Higher probability = more likely next token</i>"]

    %% Token Selection
    J --> K["🎯 DECODING PIPELINE (PRECISE ORDERING)<br/>1. Apply Temperature: logits ÷ temperature<br/>2. Filter by Top-k: keep k highest logits<br/>3. Filter by Top-p: keep tokens with cumulative prob ≤ p<br/>4. Renormalize & Sample from filtered distribution<br/>5. Stop Criteria: EOS token, max length, custom filters<br/><i>Order matters: temperature → filtering → sampling</i>"]
    K --> L["✨ Selected Token: Paris<br/>Chosen based on probability + sampling<br/><i>This becomes the next word</i>"]

    %% Autoregressive Loop
    L --> M["🔄 Updated Token IDs<br/>[1234, 567, 89, 1011, 1213, 445, 63, 8901]<br/><i>Append new token ID to sequence</i>"]
    M -.->|"🔁 AUTOREGRESSIVE LOOP<br/>Feed back token IDs for next prediction<br/>Repeat until EOS or max length"| C

    %% Generation Example
    subgraph Generation["🔄 AUTOREGRESSIVE GENERATION PROCESS"]
        N1["Step 1: What is the capital of France?"] --> N2["→ Predict: Paris (73% prob)"]
        N2 --> N3["Step 2: What is the capital of France? Paris"] --> N4["→ Predict: is (45% prob)"]
        N4 --> N5["Step 3: ...France? Paris is"] --> N6["→ Predict: a (23% prob)"]
        N6 --> N7["Step 4: ...Paris is a"] --> N8["→ Predict: beautiful (18% prob)"]
        N8 --> N9["Step 5: Continue until <EOS> or max length..."]
    end

    %% Attention Visualization
    subgraph AttentionViz["🎯 SCALED DOT-PRODUCT ATTENTION (EXACT FORMULA)"]
        O1["Query Matrix Q<br/>[seq_len × d_k]"] --> O2["Compute QK^T<br/>Raw attention scores<br/>[seq_len × seq_len]"]
        O2 --> O6["Scale by √d_k<br/>QK^T/√d_k<br/>Prevents vanishing gradients when d_k is large"]
        O6 --> O4["Apply Causal Mask<br/>Set upper triangular to -∞<br/>Then softmax → attention weights"]
        O4 --> O3["Multiply by Values<br/>softmax(QK^T/√d_k + mask) × V<br/>Final context-aware representations"]
    end

    %% Styling
    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef processing fill:#e1f5fe,stroke:#039be5,stroke-width:2px
    classDef attention fill:#f3e5f5,stroke:#8e24aa,stroke-width:3px
    classDef output fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef generation fill:#fff3e0,stroke:#f57c00,stroke-width:2px

    class A,B,C,D,E input
    class DecoderStack,F6,G0,G2,G3,G4,G5 processing
    class G1,O1,O2,O3,O4,O6 attention
    class H,I,J,K,L,M output
    class N1,N2,N3,N4,N5,N6,N7,N8,N9 generation
