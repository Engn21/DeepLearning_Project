# DeepLearning_Project
This project is a character-level GPT (Generative Pre-trained Transformer) language model written using the PyTorch by ENGIN SAMET DEDE and SEVGI DILAY DEMIRCI. The model was trained on the Tiny Shakespeare dataset provided in the assignment and aims to generate short Shakespeare-like texts using the texts found in this dataset.  

The project aims to demonstrate the concepts of modern language models taught in specific courses in a practical way. These are important concepts used in the project, such as Transformer architecture, Causal Self-Attention, Residual Connections, Layer Normalization, and Next-Token Prediction.

## The project's objectives can be explained by the following principles: 

-Initially, we mainly aimed to understand and implement a Transformer-based GPT architecture properly as much as possible.

-The other objective is to learn the concept of language modeling at the character level.

-Also seeing how the Self-Attention mechanism works in practice was also important.

-Finally experiencing the text generation process from start to finish was beneficial for us to understand the course concept better.

## Data Set Used

Tiny Shakespeare Dataset

Source:

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

This data set consists of approximately 1 MB of plain text from Shakespeare's writings.
## Model Architecture

The model is based on the decoder-only Transformer (GPT) architecture. The structure can be better explained using below order:
```
Input Characters
      ↓     
Token Embedding + Position Embedding

      ↓    
[ Transformer Block × 12 ]

      ↓      
LayerNorm

      ↓      
Linear (Vocabulary Projection)
      ↓      
Next Character Prediction
```
### Input Characters

Since the inputs here are in string form, tokenization is performed, and tokenization is achieved at the character level. When considering a sentence like “O God, O God!”, all these letters and spaces correspond to a number in a pre-built dictionary (stoi).

In example:

'O' -> 34

' ' -> 0

'G' -> 18

At this stage, the model still knows nothing meaningful; tokens have only been converted into numbers.

### Token Embedding + Position Embedding


This is the most important step where the model gets to learn raw character information and turn this into meaningful numeric representations that we shall utilize in future implementations. Transformer architectures do not accept inputs in the form of characters or numbers; thus, inputs are previously converted to high-dimensional vectors.

The contribution to the model in this project is a character IDs-based tensor. Yet, these IDs are just indexes but they do not have any semantic meaning themselves. Thus, the first step is the Token Embedding process.

During the token embedding phase, the identity of all characters is translated into a dense n_embd = 768-dimensional vectors of the nn.Embedding layer. Through this process, the model will be able to generate a learnable representation of every character. The characters with similar contexts drift towards one another in the embedding space over time. The model is based on learning relationships between characters and this process.

This is applied in the code line below:

`tok_emb = self.wte(idx)`

 In this case, idx is a character ID tensor of size (B, T) and the result of this line is an embedding tensor of size (B, T, n_embd).
Nevertheless, it is not enough to use token embedding only. Transformer architectures lack knowledge of order information. That is, although the model may have the same characters in different sequences it is not necessarily perceived so. This problem is solved by Position Embedding.

During position embedding, an additional embedding vector which depicts the location of each character in the sequence is produced. These vectors give the model the position of the character in the sentence. In such a way, the model does not only learn information of the type of which character, but also of the type, where this character is information.

This is executed in the code as below:

`pos = torch.arange(0, T, dtype=torch.long, device=device)`
`pos_emb = self.wpe(pos)`

In this case, pos is an array of position indices (0 to T-1) and self.wpe is an embedding layer that maps each position to a learnable n_embd-dimensional vector.
Once, the token embedding and the position embedding processes have been performed, these two vectors are added as a single representation:

`x = self.drop(tok_emb + pos_emb)`

Through this addition, the resulting representation formed of every token contains both the identity of the character and its place in the order. This is followed by the dropout in order to stop overfitting.

Consequently, by the completion of this phase, the model is presented with a more highly informed input representation that can be used to learn context, to be fed to the Transformer blocks.

## Transformer Blocks
After the token embedding and position embedding, we should also observe that the model has numerical representations for each character that contain identity and position information both. However, these representations are still considered as context-independent. 

In other words, at this point, the model has not yet learned how the characters relate to each other. The structure that has been utilized to learn this relationship is the Transformer Blocks that is used sequentially in the model. In this project, a total of 12 Transformer Blocks are stacked on top of each other in identical structures. Each block processes the input a little more to produce a richer and more contextual representation. This structure is defined in the code as follows: 

`self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])`

and in the forward function:

`for block in self.blocks: x = block(x)`

Thanks to this loop, the same type of Transformer Block is applied 12 times in a row.

### Main Purpose of Transformer Blocks

The fundamental purpose of each Transformer Block is to learn the relationship between each character and the characters that precede it, and to reflect this information in its representation.

This enables the model to learn:

-Which characters frequently appear together

-Which characters gain meaning in which contexts

-Long-range dependencies (For example, the relationship between the beginning and end of a sentence)

Each Transformer Block consists of two main operations: Causal Self-Attention and Feed Forward Network (MLP). These two operations are supported by Layer Normalization and Residual Connections.

## Causal Self-Attention: 
Causal Self-Attention is the core mechanism that allows our GPT model to understand context and relationships between characters in a sequence. This section explains how it works in our implementation, connecting the mathematical foundations directly to the code.

Within each Transformer Block, the first major operation involves normalizing the input through Layer Normalization and then passing it through the Causal Self-Attention layer. In our code, this entire flow is captured in a single line:
```python
x = x + self.attn(self.ln1(x))
```

Let's break down what this line actually does. First, the current representation `x` is normalized via `self.ln1` to provide a more stable input for the attention mechanism. Then, the normalized input passes through `self.attn(...)`, which gathers contextual information from previous positions in the sequence. Finally, this newly computed information is added back to the original representation through a residual connection, preventing information loss as data flows through the network.

---

The tensor `x` entering this line contains the combined token and positional embeddings. Its shape is $x \in \mathbb{R}^{B \times T \times C}$, where B is the batch size (how many sequences we process in parallel), T is the sequence length (up to `block_size`, which is 128 in our case), and C is the embedding dimension (`n_embd = 768`). In practical terms, each token (character) in our sequence is represented by a 768-dimensional vector that encodes both its identity and its position within the sequence.

---

Before the attention computation begins, the input passes through Layer Normalization:
```python
self.ln1 = nn.LayerNorm(n_embd)
# ...
self.ln1(x)
```

LayerNorm performs mean-variance normalization independently on each token's embedding vector. For a single token vector $x_{b,t,:}$, the computation is:

$$\text{LN}(x_{b,t,:}) = \gamma \odot \frac{x_{b,t,:} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Here, $\mu$ and $\sigma^2$ are computed across the feature (embedding) dimension of that specific token. The learnable parameters $\gamma$ and $\beta$ allow the model to adjust the normalized output as needed. The key insight is that this normalization happens before the attention operation, not after. This architectural choice, known as the "Pre-LN Transformer" design, provides better training stability compared to the original "Post-LN" approach. It ensures that the values entering the attention mechanism have a consistent scale, which helps prevent numerical instabilities during training.

---

Once the normalized tensor enters the attention layer, the first operation is to produce three different projections for each token: Query (Q), Key (K), and Value (V). In our code:
```python
qkv = self.c_attn(x)                 # Linear: (B, T, C) -> (B, T, 3C)
q, k, v = qkv.split(self.n_embd, dim=2)
```

This is implemented efficiently as a single linear transformation that outputs a tensor three times the embedding size, which is then split into three equal parts. Mathematically, this corresponds to

$[Q \; K \; V] = XW + b$,

and after splitting:

$Q = XW_Q$, $K = XW_K$, $V = XW_V$,

where $X$ is the LayerNorm output, and $W_Q$, $W_K$, $W_V$ are the learned weight matrices (conceptually separate, but stored together in `self.c_attn` for computational efficiency).

What do Q, K, V represent intuitively? Query (Q) represents "what information am I looking for?" for each token. Key (K) represents "what information do I contain?" for each token. Value (V) represents "what information will I contribute if selected?" for each token. The attention mechanism works by comparing queries against keys to determine relevance, then using those relevance scores to weight the values.

---

Our model uses 8 attention heads (`n_head = 8`), which means Q, K, and V are split into 8 separate subspaces. The code accomplishes this through reshape and transpose operations:
```python
k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
```

After this transformation, the shapes become $Q, K, V \in \mathbb{R}^{B \times h \times T \times d_h}$, where $h = 8$ (number of heads) and $d_h = C / h = 768 / 8 = 96$ (the dimension per head). We are using multiple heads. To better understand it can be explained with such example as, in language modeling, one head might focus on syntactic relationships (subject-verb agreement), another on semantic relationships (word meanings in context), and yet another on local character patterns. By running 8 such attention computations in parallel and combining their results, the model gains a much richer understanding of the input than a single attention mechanism could provide.

---

Now comes the core of the attention mechanism: computing how much each token should attend to every other token. This is done through a scaled dot-product:
```python
att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
```

Mathematically, this is $A = \frac{QK^T}{\sqrt{d_h}}$. The matrix multiplication `q @ k.transpose(-2, -1)` computes the dot product between every query and every key, resulting in a $T \times T$ matrix of raw attention scores for each head and each batch element. Why divide by $\sqrt{d_h}$? This scaling factor is crucial for training stability. Without it, when the head dimension $d_h$ is large, the dot products can grow very large in magnitude. Large values cause the subsequent softmax operation to produce extremely peaked distributions (essentially one-hot vectors), which leads to vanishing gradients. By scaling down by $\sqrt{d_h}$, we keep the dot products in a range where softmax produces smoother, more trainable distributions.

---

In language modeling, a fundamental constraint is that when predicting the next character, the model cannot "see" future characters. This is enforced through a causal mask:
```python
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
```

The mask itself is created during initialization:
```python
self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))
```

This creates a lower-triangular matrix of ones. The `masked_fill` operation then sets all positions where the mask is zero (i.e., the upper triangle, representing future positions) to negative infinity. Mathematically, the masked attention scores become:

$$A_{i,j} = \begin{cases} A_{i,j} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

For position $i$, only positions $j \leq i$ (the current and all previous positions) remain visible. Setting future positions to $-\infty$ ensures they become exactly zero after the softmax operation, effectively making them invisible to the model. For example, when the model is processing the 5th character in a sequence, it can only attend to characters 1, 2, 3, 4, and 5. Characters 6, 7, 8, etc. are completely hidden, as if they don't exist.

---

After masking, each row of the attention matrix is passed through a softmax function:
```python
att = F.softmax(att, dim=-1)
```

Mathematically, this is $\alpha_{i,j} = \frac{e^{A_{i,j}}}{\sum_{k \leq i} e^{A_{i,k}}}$. This transforms the raw scores into a probability distribution. The resulting $\alpha_{i,j}$ values represent the answer to the question: "When processing token $i$, how much attention should be paid to token $j$?" Since we're dividing by the sum of exponentials only over valid positions ($k \leq i$), the attention weights for each position sum to 1. The $-\infty$ values from the causal mask become exactly 0 after softmax, contributing nothing to the weighted sum.

---

With the attention weights computed, we now gather information by taking a weighted sum of the value vectors:
```python
y = att @ v  # (B, n_head, T, head_size)
```

Mathematically, for each position $i$: $y_i = \sum_{j \leq i} \alpha_{i,j} V_j$. This is the heart of what attention accomplishes. Each token's new representation is a weighted combination of all the tokens it can see, where the weights reflect the learned relevance. A token that is highly relevant to the current position contributes more to the output; irrelevant tokens contribute less. After this operation, each token's representation has been enriched with contextual information from its past. The model has effectively learned to "look back" and gather relevant information.

---

The outputs from all 8 attention heads need to be combined back into a single representation:
```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
return self.resid_dropout(self.c_proj(y))
```

This concatenates the head outputs and applies a final linear projection. Mathematically, this is $Y = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$. The output projection $W_O$ allows the model to learn how to best combine the information gathered by the different heads. The dropout applied here (`resid_dropout`) helps prevent overfitting by randomly zeroing some elements during training.

---

Returning to the full Block-level view with `x = x + self.attn(self.ln1(x))`, the residual (skip) connection adds the attention output to the original input. Mathematically, this is $H' = H + \text{Attn}(\text{LN}(H))$. This seemingly simple addition is actually critical to the success of deep transformer models. The residual connection serves two important purposes. First, information preservation: the attention mechanism adds new contextual information without overwriting what was already there, so the original token identity and positional information remain accessible to later layers. Second, gradient flow: during backpropagation, gradients can flow directly through the addition operation, bypassing the attention computation entirely, which creates a "gradient highway" that allows training signals to reach early layers without vanishing. This is essential for training networks as deep as our 12-layer model. Without residual connections, training deep transformers would be practically impossible because the gradients would either vanish (becoming too small to drive learning) or explode (becoming so large they destabilize training) long before reaching the early layers.

---

In summary, the Causal Self-Attention mechanism transforms each token's representation by allowing it to gather relevant information from all previous positions in the sequence. Through the interplay of queries, keys, and values across multiple heads, the model learns which past tokens are relevant for predicting what comes next. The causal mask ensures the model never "cheats" by looking at future tokens, which is essential for the autoregressive language modeling task. Combined with Layer Normalization for stability and residual connections for trainability, this mechanism forms the foundation of our GPT's ability to generate coherent, Shakespeare-like text.


#### Layer Normalization (LayerNorm)

Layer Normalization is a normalization technique used to stabilize the scale and distribution of inputs to a neural network layer.  
In essence, it shifts the mean of a vector toward zero and its variance toward one.  
This stabilization is critical in deep architectures such as Transformers.

In deep Transformer models:

- Some neurons produce very large activations  
- Others produce very small activations  
- These imbalances compound across layers  

As a result:

- Training becomes unstable  
- Gradients may explode or vanish  
- Learning becomes inefficient or fails  

Layer Normalization mitigates these issues by enforcing consistent activation statistics.

#### Mathematical Definition

Consider the embedding vector of a single token:
```text
x = [2.4, -1.2, 0.7, 5.9, -3.1, ...]  (768 dimensions)
```

Layer Normalization computes:

- The mean $\mu$ over the feature dimension
- The variance $\sigma^2$ over the feature dimension

Each element is normalized as:

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

Two learnable parameters are then applied:

- $\gamma$ (scale)
- $\beta$ (shift)

Resulting in:

$$y_i = \gamma \hat{x}_i + \beta$$

LayerNorm does not remove information; it only regulates the scale of representations.


Layer Normalization operates on a single token and across the embedding (feature) dimension.

It does not depend on batch statistics and other tokens in the sequence

This makes it particularly suitable for Transformer-based NLP models where batch sizes and sequence lengths may vary.

#### Comparison with Batch Normalization

| Batch Normalization | Layer Normalization |
|---------------------|---------------------|
| Depends on batch statistics | Independent of batch |
| Common in computer vision | Standard in NLP |
| Sensitive to batch size | Batch size invariant |

Transformers therefore rely on Layer Normalization.

#### Usage in Transformer Architecture in the code

Layer Normalization appears in three locations.

##### Pre-Attention Normalization
```python
self.ln1 = nn.LayerNorm(n_embd)
x = x + self.attn(self.ln1(x))
```

Purpose:

- Stabilize inputs before attention
- Prevent excessively large attention scores

##### Pre-MLP Normalization
```python
self.ln2 = nn.LayerNorm(n_embd)
x = x + self.mlp(self.ln2(x))
```

Purpose:

- Provide stable inputs to the MLP
- Improve behavior of non-linear transformations

##### Final Layer Normalization
```python
self.ln_f = nn.LayerNorm(n_embd)
x = self.ln_f(x)
```

Purpose:

- Rebalance representations after all Transformer blocks
- Stabilize the input to the final linear projection
- Improve logit distribution stability

#### Repeated Application of LayerNorm

In deep models, normalization must be maintained continuously.

- Block-level LayerNorm ensures local stability
- Final LayerNorm ensures global stability

This repetition is a deliberate architectural choice in Transformer models.

#### Pre-LayerNorm Transformer

The formulation:
```python
x = x + self.attn(self.ln1(x))
```

corresponds to a Pre-LayerNorm Transformer, where normalization is applied before each sub-layer.

Advantages:

- Improved training stability
- Better gradient flow
- Enables deeper architectures

This design is standard in GPT-2 and later models.

To sum up, Layer Normalization normalizes token embeddings across the feature dimension, enabling stable and efficient training of deep Transformer architectures.

---

### Residual Add (Residual Connection / Skip Connection)

Residual Add is a structural mechanism that adds a layer's output to its input, preserving information flow across deep networks.


Residual connections are defined as:

$$\text{Output} = x + F(x)$$

Where:

- $x$ is the input
- $F(x)$ is the transformation applied by the layer (e.g., attention or MLP)

This allows the model to augment representations without overwriting existing information. In deep neural networks without residual connections:

- Information degrades across layers
- Gradients weaken during backpropagation
- Training deep models becomes impractical

This phenomenon is known as the vanishing gradient problem.


Residual connections introduce shortcut paths that:

- Preserve information
- Enable direct gradient flow
- Make deep architectures trainable

Each Transformer block contains two major transformations:

1. Self-Attention
2. Feed Forward Network (MLP)

Both are powerful but potentially destabilizing.

Residual connections allow these transformations to be applied additively rather than destructively.

#### Implementation in Transformer Blocks

##### After Self-Attention
```python
x = x + self.attn(self.ln1(x))
```

The attention output is added to the original input.

##### After MLP
```python
x = x + self.mlp(self.ln2(x))
```

The transformed representation is added while preserving prior information.

#### Information Preservation

Residual connections do not simply copy inputs.

They accumulate information progressively:

- Attention adds contextual information
- MLP adds feature transformations
- Residual connections preserve and combine both

#### Interaction with Layer Normalization

Residual connections and Layer Normalization are complementary:

- Residual connections preserve information and gradients
- Layer Normalization stabilizes magnitude and scale

Using one without the other leads to instability.

#### Transformer Block Flow

Therefore the procedure can be summarized as below:
```
Input
 ↓
LayerNorm
 ↓
Causal Self-Attention
 ↓
Residual Add
 ↓
LayerNorm
 ↓
Feed Forward (MLP)
 ↓
Residual Add
 ↓
Output
```
#### Feed Forward Network (MLP) Layer

The representation coming out of the Causal Self-Attention stage now contains contextual information gathered from previous tokens for each position. At this point, the model has learned the relationships between tokens, but this information has not yet been deeply processed. The goal now is to enrich this contextual information through non-linear transformations, and this is exactly what the Feed Forward Network (MLP) accomplishes.

This is why every Transformer Block contains an MLP layer right after the attention mechanism. In our code, this transition is captured in a single, clear line:
```python
x = x + self.mlp(self.ln2(x))
```

This line summarizes everything that happens in the MLP stage. The current representation `x` first passes through Layer Normalization, then gets processed by the MLP, and finally gets added back to the original representation through a residual connection. This design is a deliberate and standard choice in Transformer architectures.

---

The tensor `x` entering the MLP has shape $B \times T \times C$. At this point, each token no longer carries just "which character it is" but also "how it relates to previous characters" thanks to the attention mechanism. However, this information consists of linear combinations. The fundamental purpose of the MLP is to transform this information non-linearly to obtain a more abstract and powerful representation.

---

Before entering the MLP, Layer Normalization is applied to stabilize the distribution that formed after attention and residual addition. Mathematically, LayerNorm normalizes each token vector across its embedding dimension. This step ensures numerical stability at the MLP input and makes stable training possible, especially in deep Transformer architectures. In the code, this normalization is done with:
```python
self.ln2 = nn.LayerNorm(n_embd)
```

---

The vectors coming out of LayerNorm then enter the first linear layer of the MLP. This layer temporarily expands the embedding dimension by a factor of four. In other words, for each token, a vector of dimension $C$ is transformed into a vector of dimension $4C$. Mathematically, this operation is:

$$h = xW_1 + b_1$$

The purpose of this expansion is to enlarge the representation space so the model can learn more complex patterns. This "expand first, then contract" approach is standard in Transformer architectures. By projecting into a higher-dimensional space, the network gains the capacity to represent more intricate functions and relationships that would be impossible to capture in the original dimension.

---

This expanded representation then passes through the GELU activation function. GELU provides a non-linearity that softly suppresses small and uncertain signals while allowing strong signals to pass through. This enables the model to transform the information from attention in a more flexible and meaningful way. In the code, this step is clearly visible:
```python
nn.GELU()
```

Why GELU instead of ReLU? Unlike ReLU which has a hard cutoff at zero, GELU provides a smooth transition. This means that slightly negative values are not completely zeroed out but are dampened proportionally. This smoother behavior leads to better gradient flow during training and has been shown to work particularly well in Transformer-based language models.

---

The representation coming out of GELU is then reduced back to the original embedding dimension in the second linear layer of the MLP. This operation is mathematically expressed as:

$$\text{MLP}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

The purpose here is to ensure the MLP output has the appropriate dimension to be added to the input through the residual connection. The complete MLP structure in the code looks like this:
```python
class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

---

After this transformation, dropout is applied. Dropout randomly deactivates some neurons during training, preventing the model from memorizing specific pathways. This step makes the MLP's learning process more generalizable. In the code, this is a natural part of the MLP block:
```python
nn.Dropout(dropout)
```

With a dropout rate of 0.1 in our implementation, 10% of the neurons are randomly set to zero during each training step. This forces the network to develop redundant representations, ensuring that no single neuron becomes too critical for the final output.

---

Finally, the output produced by the MLP is added to the post-attention representation through the residual connection. This operation reflects one of the most important principles of the Transformer architecture: new information is added, old information is not erased. Mathematically, this step is expressed as:

$$H^{(l+1)} = H' + \text{MLP}(\text{LN}(H'))$$

where $H'$ is the post-attention representation.

---

Through this design, the model progressively enriches information in each Transformer Block. While attention learns the relationships between tokens, the MLP processes the information extracted from these relationships, abstracts it, and increases the representational power. The residual connections and Layer Normalization ensure that this process continues without degradation even in deep architectures.

The division of labor between attention and MLP is worth emphasizing: attention is responsible for communication between tokens (deciding which tokens should influence which), while the MLP is responsible for computation within each token (transforming the gathered information into more useful features). Together, they form the complete Transformer Block that gets repeated 12 times in our model.
### Why 12 Transformer blocks?

A single Transformer Block can learn context to a limited extent. However, as blocks are stacked:

- **Lower layers** → learn more local relationships
- **Middle layers** → capture syntactic structures
- **Top layers** → learn more abstract and semantic relationships

The 12-layer structure used in this project is compatible with the GPT-2 architecture and ensures that the model achieves sufficient contextual depth at the character level.

---

### End of Transformers Block

At the end of the [Transformer Block × 12] phase, each character representation in the model's possession becomes:

- Aware of previous characters
- Context-aware
- Rich enough to predict the next character

This output is now ready to be sent to the final prediction layer (Final LayerNorm and Linear Projection).

---

## Final Output Layer

After the representations pass through all 12 Transformer Blocks, the model has learned rich, context-aware representations for each character position. However, these representations are still in the form of 768-dimensional vectors. To make actual predictions about which character should come next, we need to convert these high-dimensional vectors back into probabilities over our vocabulary of 65 characters. This is accomplished by the final output layer, which consists of two components: Final Layer Normalization and the Language Modeling Head.

### Final Layer Normalization

Before making the final prediction, one last Layer Normalization is applied to the output of all Transformer Blocks:

```python
self.ln_f = nn.LayerNorm(n_embd)
x = self.ln_f(x)
```

This final normalization serves several critical purposes:

- **Stabilizes the final representations**: After passing through 12 Transformer Blocks with multiple residual additions, the scale of the representations can drift. The final LayerNorm rebalances them to a consistent scale.
- **Improves logit distribution**: By normalizing before the final linear projection, we ensure that the output logits (raw prediction scores) have a reasonable range, which leads to better-behaved probability distributions after softmax.
- **Enhances training stability**: This normalization helps prevent numerical instabilities that could arise from extreme values entering the final projection layer.

The mathematical operation remains the same as in previous LayerNorm applications:

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu$ and $\sigma^2$ are computed over the embedding dimension for each token independently.

---

### Language Modeling Head (Linear Projection)

The Language Modeling Head is a linear layer that projects the normalized 768-dimensional representations back to the vocabulary size (65 characters in our case):

```python
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
logits = self.lm_head(x)
```

Mathematically, this operation is:

$$\text{logits} = xW^T$$

where $W \in \mathbb{R}^{vocab\_size \times n\_embd}$ is the weight matrix, and logits $\in \mathbb{R}^{B \times T \times vocab\_size}$.

For each position in the sequence, the model produces a vector of 65 numbers (one for each character in our vocabulary). These numbers, called logits, represent the model's raw, unnormalized predictions about which character should come next.

---

### Weight Tying

An important implementation detail in our model is weight tying between the token embedding layer and the language modeling head:

```python
self.wte.weight = self.lm_head.weight
```

This means that the same weight matrix is used for both:
1. Converting character IDs to embeddings (token embedding)
2. Converting final representations back to character predictions (language modeling head)

**Why use weight tying?**

- **Parameter efficiency**: Instead of learning two separate large matrices, we share the weights, reducing the total parameter count by approximately 50,000 parameters (65 × 768).
- **Improved generalization**: This constraint forces the model to learn a unified representation space where the embedding of a character and its prediction weights are related, which can improve the model's ability to generalize.
- **Theoretical motivation**: If a character's embedding represents "what this character means," it makes sense that the same vector should also represent "how likely this character is as an output."

This technique is standard in modern language models and has been shown to improve performance, especially on smaller datasets like Tiny Shakespeare.

---

## Training Process

The model is trained using standard supervised learning techniques for language modeling. The training process involves feeding sequences of characters to the model and teaching it to predict the next character at each position.

### Loss Function: Cross-Entropy Loss

The model uses Cross-Entropy Loss to measure how well its predictions match the actual next characters. Cross-entropy is the standard loss function for classification tasks, and language modeling can be viewed as a classification problem where, at each position, we're classifying which of the 65 characters should come next.

Given the model's logits (raw predictions) and the target character IDs, the loss is computed as follows:

```python
loss = F.cross_entropy(logits, targets)
```

Before computing the loss, both logits and targets are reshaped:

```python
B, T, C = logits.shape  # B=batch size, T=sequence length, C=vocab size
logits = logits.view(B*T, C)  # Reshape to (B*T, C)
targets = targets.view(B*T)    # Reshape to (B*T)
```

**Why reshape?** PyTorch's cross-entropy function expects:
- Predictions: a 2D tensor of shape (N, C) where N is the number of samples and C is the number of classes
- Targets: a 1D tensor of shape (N) containing the correct class indices

By reshaping from (B, T, C) to (B×T, C), we're treating each position in each sequence as an independent prediction problem. For example, with batch size 64 and sequence length 128, we're making 64 × 128 = 8,192 predictions simultaneously.

**Mathematical formulation:**

For a single prediction at position $i$ with target class $y_i$, the cross-entropy loss is:

$$L_i = -\log\left(\frac{e^{z_{y_i}}}{\sum_{j=1}^{C} e^{z_j}}\right) = -z_{y_i} + \log\left(\sum_{j=1}^{C} e^{z_j}\right)$$

where $z$ represents the logits. The total loss is the average over all positions:

$$L = \frac{1}{B \times T} \sum_{i=1}^{B \times T} L_i$$

**Intuition:** Cross-entropy loss is minimized when the model assigns high probability to the correct next character and low probability to all other characters. A perfect model would have a loss of 0, while a random model would have a loss of approximately $\log(65) \approx 4.17$ for our vocabulary size.

---

### Optimization: AdamW

The model uses the AdamW optimizer to update its parameters during training:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

**AdamW (Adam with Weight Decay)** is an improved version of the popular Adam optimizer that:

- Adapts the learning rate for each parameter based on first and second moments of gradients
- Applies weight decay (L2 regularization) in a more principled way than standard Adam
- Is particularly effective for training Transformer models

**Learning rate:** The learning rate of 3e-4 (0.0003) is a commonly used value for training GPT-style models. This rate was chosen because:
- It's large enough to make meaningful progress during training
- It's small enough to avoid instability in deep 12-layer architectures
- It's a standard baseline from GPT-2 and other Transformer language models

---

### Training Loop

The training process follows these steps for each iteration:

1. **Fetch a batch of data**: Get a batch of input-output pairs (xb, yb) from the dataset
2. **Forward pass**: Pass the inputs through the model to get predictions and compute loss
3. **Backward pass**: Compute gradients of the loss with respect to all model parameters
4. **Parameter update**: Update the model weights using the AdamW optimizer
5. **Monitor progress**: Print the loss periodically to track training progress

```python
for iter_num in range(max_iters):
    xb, yb = next(data_iter)  # Get batch
    xb, yb = xb.to(device), yb.to(device)  # Move to GPU/CPU

    logits, loss = model(xb, yb)  # Forward pass

    optimizer.zero_grad(set_to_none=True)  # Clear old gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters

    if iter_num % 100 == 0:
        print(f"Step {iter_num}: Loss {loss.item():.4f}")
```

**Key implementation details:**

- `optimizer.zero_grad(set_to_none=True)`: Clears gradients from the previous iteration. The `set_to_none=True` argument is a performance optimization that sets gradients to None instead of zeroing them, which is slightly faster.
- `loss.backward()`: Computes gradients using backpropagation through the entire 12-layer Transformer architecture.
- `optimizer.step()`: Updates all ~85 million parameters based on the computed gradients.

---

### Hyperparameter Experimentation

During the development of this project, extensive hyperparameter tuning was performed using **Kaggle notebooks** to find the optimal configuration. Different values were tested for:

- **max_iters**: Training iterations (tested: 3000, 6000, 8000+)
- **learning_rate**: Learning rate values (tested various rates around 3e-4)
- **batch_size**: Originally 64, experimented with different sizes based on available GPU memory
- **n_layer**: Number of Transformer blocks (settled on 12 for GPT-2-like architecture)
- **dropout**: Dropout rate for regularization (tested different values around 0.1)

**Note on max_iters:** The hyperparameter values shown in the code (such as `max_iters = 6000`) represent one of several configurations tested. The actual training runs varied, with some experiments running for 8000+ iterations to observe convergence behavior. The Kaggle environment provided GPU acceleration which made these experiments feasible.

**Typical training progression:**

- **Initial Loss (~4.35)**: At the beginning, the model is essentially guessing randomly, resulting in high loss close to $\log(65) \approx 4.17$
- **Early Phase (0-2000 steps)**: Rapid loss decrease as the model learns basic character frequencies and common patterns
- **Middle Phase (2000-5000 steps)**: Steady improvement as the model learns word structures and common phrases
- **Late Phase (5000+ steps)**: Slower refinement as the model learns more complex dependencies and Shakespeare's writing style
- **Final Loss (~0.33-0.67)**: The final loss varies depending on training duration, indicating the model has learned substantial structure in the data

The loss curve typically shows smooth, monotonic improvement with occasional small fluctuations due to the stochastic nature of mini-batch training.

---

## Text Generation

After training, the model can generate new Shakespeare-like text through an autoregressive generation process. This means the model generates one character at a time, using its own previous predictions as context for the next prediction.

### Autoregressive Generation Process

The generation process is implemented in the `generate` method:

```python
def generate(self, idx, max_new_tokens, temperature=0.7):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits / temperature, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

**Step-by-step breakdown:**

1. **Context windowing** (`idx_cond = idx[:, -block_size:]`):
   - Takes only the last `block_size` (128) characters as context
   - This ensures the context length doesn't exceed what the model was trained on
   - If the generated sequence becomes longer than 128 characters, we use a sliding window

2. **Forward pass** (`logits, _ = self(idx_cond)`):
   - Passes the context through the entire model
   - Gets predictions for all positions in the context
   - We only need the prediction for the last position

3. **Extract last position** (`logits = logits[:, -1, :]`):
   - Takes the logits for the final position only
   - This gives us the model's prediction for what character should come next
   - Shape: (B, vocab_size) = (1, 65) for single sequence generation

4. **Apply temperature and softmax** (`probs = F.softmax(logits / temperature, dim=-1)`):
   - Converts logits to probabilities using softmax with temperature scaling
   - Temperature controls the randomness of predictions (explained below)

5. **Sample next token** (`idx_next = torch.multinomial(probs, num_samples=1)`):
   - Randomly samples one character from the probability distribution
   - Higher probability characters are more likely to be selected, but lower probability ones can still be chosen

6. **Append to sequence** (`idx = torch.cat((idx, idx_next), dim=1)`):
   - Adds the newly generated character to the sequence
   - This extended sequence becomes the context for generating the next character

7. **Repeat**: Steps 1-6 repeat for `max_new_tokens` times

---

### Temperature Sampling

Temperature is a hyperparameter that controls the randomness and creativity of the generated text. It works by scaling the logits before applying softmax:

```python
probs = F.softmax(logits / temperature, dim=-1)
```

**Mathematical effect:**

- **Temperature = 1.0**: Standard softmax, uses the model's original probability distribution
- **Temperature < 1.0** (e.g., 0.5): Makes the distribution sharper (more peaked)
  - High-probability characters become even more likely
  - Low-probability characters become even less likely
  - Results in more conservative, predictable text
- **Temperature > 1.0** (e.g., 1.5): Makes the distribution flatter (more uniform)
  - Probabilities become more evenly distributed
  - Results in more creative, diverse, but potentially less coherent text

**Our choice: temperature = 0.7**

We use a temperature of 0.7, which is slightly lower than 1.0. This choice:
- Maintains most of the model's learned probability distribution
- Slightly favors higher-probability predictions for better coherence
- Still allows for creative variation and diversity in the output
- Balances between deterministic (boring) and random (incoherent) generation

**Example:** If the model predicts the next character with probabilities:
- 'e': 0.6, 't': 0.25, 'h': 0.10, 'a': 0.05

With temperature = 0.7, these might become:
- 'e': 0.68, 't': 0.22, 'h': 0.07, 'a': 0.03

This makes the most likely character ('e') even more likely to be selected, while still preserving some chance for the other characters.

---

### Inference Process

To generate text, the model requires an initial context (prompt):

```python
# Tokenization: Convert string to character IDs
def tokenize(s):
    return torch.tensor([train_dataset.stoi[c] for c in s],
                        dtype=torch.long, device=device).unsqueeze(0)

# Detokenization: Convert character IDs back to string
def tokens_to_string(tokens):
    return ''.join([train_dataset.itos[i.item()] for i in tokens[0]])

# Set model to evaluation mode
model.eval()
with torch.no_grad():
    context_str = "O God, O God!"
    tokenized_context = tokenize(context_str)

    # Generate 500 new characters
    y = model.generate(tokenized_context, max_new_tokens=500)

    # Convert back to readable text
    completion = tokens_to_string(y)
    print(completion)
```

**Key points:**

- `model.eval()`: Switches the model to evaluation mode, which disables dropout and ensures batch normalization uses running statistics
- `torch.no_grad()`: Disables gradient computation for faster inference and reduced memory usage
- The model takes the initial context "O God, O God!" and generates 500 additional characters
- The output includes both the original context and the newly generated text

---

### Sample Generated Outputs

During experimentation on Kaggle, multiple model configurations were tested with different architectures and hyperparameters. Below are sample outputs from two different model configurations, demonstrating how model size and training duration affect text generation quality.

---

#### Output 1: Smaller Model Configuration (10.79M Parameters)

**Model Configuration:**
- Parameters: ~10.79 million
- Training iterations: 5,000 steps
- Final Loss: 1.1439
- Sampling method: Top-K sampling (K=10)

**Generated Text:**

```
I'll all the deputy, but it on the which is thought the
good that I strike to thee, for thy bed, that art tooth
without sir, who cames to thee, though the time to do
now-may suffice fick unto myself?

BUSHY:
Troth, mighty took marry fortune former, thou shadst
him thy fitteed torth: what, monthough I want,
and no looked at look more but in a merile?

SLY:
Say tell, that, I will stand bid my father's faceless,
And nothing for your lead scorn with their death,
And back it to down with their barry
```

**Analysis:**

- The model successfully learned the dialogue format with character names (BUSHY, SLY)
- It captures some Shakespearean vocabulary ("thee," "thy," "thou")
- However, there are more frequent non-words and grammatical errors ("fick," "fitteed torth," "merile")
- Sentence structure is less coherent compared to larger models
- The model shows understanding of Early Modern English patterns but struggles with longer-range coherence
- Final loss of 1.1439 indicates the model still has room for improvement

This smaller configuration demonstrates that even with fewer parameters, the model can learn basic structure, but requires more parameters and training for higher quality output.

---

#### Output 2: Larger Model Configuration (85.20M Parameters)

**Model Configuration:**
- Parameters: ~85.20 million
- Training iterations: 7,900 steps
- Final Loss: 0.3297
- Sampling method: Temperature sampling (temperature=0.7)

**Generated Text:**

```
O God, O God! what cause have I thy father?

EDWARD:
No better was a far good thing, being some careless,
by calm back to me, I go alone. What I have to say,
I will do excuse myself in person war,
And when the appetite last forced may sweet alone.

GLOUCESTER:
Now, brother Richard, will you stand by the fall?

GLOUCESTER:
My dangerous cousin, let your mother in:
I know she is come to pray forth.

LUCIO:
Ay, and a sillen bird that I live, to talk of thee,
Which triumph the fair courth of my death,
And did not
```

**Analysis:**

- Significantly better coherence and structure compared to the smaller model
- The model has learned Shakespeare's play format with proper character names (EDWARD, GLOUCESTER, LUCIO)
- Generates grammatically plausible Early Modern English structures
- Vocabulary and phrasing are more authentically Shakespeare-like ("thy father," "brother Richard," "dangerous cousin")
- Most phrases are coherent and contextually appropriate ("what cause have I," "will you stand by")
- Only occasional non-words ("courth" instead of "court"), which is expected and acceptable from a character-level model
- The model maintains excellent long-range structure with proper dialogue turns between characters
- Much lower final loss (0.3297 vs 1.1439) reflects superior learning

This larger configuration demonstrates that increasing model capacity (from 10.79M to 85.20M parameters) and training duration (from 5,000 to 7,900 steps) dramatically improves text generation quality.

---

#### Output 3: Kaggle GPU Training Run #1 (85.20M Parameters)

**Model Configuration:**
- Parameters: ~85.20 million
- Training iterations: 5,900 steps
- Final Loss: 0.6718
- Device: CUDA GPU (Kaggle)
- Sampling method: Temperature sampling (temperature=0.7)

**Generated Text:**

```
O God, O God! what, doth fregixe!
What! speak surrey, what does remember,
You may, revenge my doom at the heir,
As you'ld well believe you here.
I do ere denied to do my business
State with begging and fifty green one wood,
As she is the better sup unto me and myself
Against what will-eye I am: I say, I never
Denying the instruction of my society:
But would I revive, yet unto the house of York
Unto a best of most complain and to piece,
But let them ne'er see them measure to make an ass,
Old with Rages comman
```

**Analysis:**

- Shows moderate Shakespeare-style learning with final loss of 0.6718
- Captures some Shakespearean structures like "As you'ld well believe" and "house of York"
- Contains several invented words ("fregixe," "will-eye") characteristic of incomplete training
- Sentence structure is partially coherent but lacks smooth flow
- The model demonstrates understanding of archaic verb forms ("doth," "'ld")
- Loss value indicates the model is still in mid-training phase, having learned basic patterns but not yet achieving full coherence

---

#### Output 4: Kaggle GPU Training Run #2 (85.20M Parameters) - Best Result

**Model Configuration:**
- Parameters: ~85.20 million
- Training iterations: 5,900 steps
- Final Loss: 0.2446 (Best performance)
- Device: CUDA GPU (Kaggle)
- Sampling method: Temperature sampling (temperature=0.7)

**Generated Text:**

```
O God, O God! dishonest with this eyes
Of my mind, that boots not to be found:
I do remember what before I spoke to thee.

QUEEN MARGARET:
What, dost thou scorn me for my gentle counsel?
And soothe the devil that I warn thee from?
O, but remember this another day,
When he shall split thy very heart with sorrow,
And say poor Margaret was a prophetess.
Live each of you the subjects to his hate,
And he to yours, and all of you to God's!

HASTINGS:
My hair doth stand on end to hear her curses.

RIVERS:
And so do
```

**Analysis:**

- **Outstanding performance** - This is the best-quality output across all experiments
- Final loss of 0.2446 is significantly lower than other runs, indicating superior learning
- Generates authentic Shakespeare character names from actual plays (QUEEN MARGARET, HASTINGS, RIVERS from Richard III)
- Produces coherent, multi-line speeches that maintain thematic consistency
- The monologue by QUEEN MARGARET is remarkably coherent with proper dramatic structure
- Vocabulary is authentically Shakespearean ("dost thou scorn," "prophetess," "thy very heart")
- Grammar and syntax closely match Early Modern English
- Shows excellent understanding of dramatic dialogue with proper turn-taking between characters
- Minimal invented words - nearly all text is grammatically correct
- This result demonstrates that the same architecture can produce vastly different quality depending on training dynamics and random initialization

---

#### Output 5: Kaggle GPU Training Run #3 (85.20M Parameters)

**Model Configuration:**
- Parameters: ~85.20 million
- Training iterations: 5,900 steps
- Final Loss: 0.6972
- Device: CUDA GPU (Kaggle)
- Sampling method: Temperature sampling (temperature=0.7)

**Generated Text:**

```
O God, O God! while we think, if we have
drunk the country, we have stricken
the business of the people, and not the poor souls
Do prize their tongues to buy aside,
Your manage must make to my abode.
The soul follows to do your husband at least,
I would be quiet, or to make your patience,
If you bring me worse than all, myself,
Condemns more for than account to right.
But, soft! how canst thou bid sober?

BAPTISTA:
Is't possible you will away to-night?

PETRUCHIO:
It is too rash, too uncle, on to express the
```

**Analysis:**

- Final loss of 0.6972 indicates moderate training success
- Successfully generates recognizable Shakespeare character names (BAPTISTA and PETRUCHIO from "The Taming of the Shrew")
- Captures Shakespearean vocabulary and structures ("how canst thou," "Is't possible")
- However, some grammatical awkwardness appears ("drunk the country," "Condemns more for than account")
- Shows the model can learn character pairings from the same play
- Demonstrates variability in training outcomes even with identical configurations
- The exclamation "But, soft!" is an authentic Shakespearean expression, showing the model learned specific linguistic patterns

---

#### Comparative Insights

**Key Observations from Experimentation:**

1. **Model Size Impact:**
   - The 85.20M parameter model produces significantly more coherent and Shakespeare-like text than the 10.79M parameter model
   - Larger models can capture more complex linguistic patterns and longer-range dependencies
   - The ~8x increase in parameters (from 10.79M to 85.20M) leads to noticeably better vocabulary usage and sentence structure

2. **Training Duration and Convergence:**
   - Training for 7,900 steps vs 5,000-5,900 steps shows meaningful quality improvements
   - Extended training allows the model to refine its understanding of Shakespeare's writing style
   - The loss curve shows continued improvement beyond 5,000 steps
   - Final loss values ranging from 1.1439 (smaller model) to 0.2446 (best run) correlate directly with output quality

3. **Training Variability and Stochasticity:**
   - **Critical finding**: Multiple training runs with identical hyperparameters produced vastly different results
   - Loss values across the same 85.20M parameter model varied from 0.2446 to 0.6972
   - Output quality ranged from near-perfect Shakespeare dialogue (Output 4) to moderately coherent text (Outputs 3 and 5)
   - This demonstrates the importance of random initialization and training dynamics in deep learning
   - Multiple training runs may be necessary to achieve optimal results

4. **Loss Value as Quality Indicator:**
   - **0.24-0.33 range**: Excellent quality with authentic character names, coherent dialogue, minimal errors
   - **0.67-0.70 range**: Moderate quality with some coherence but frequent invented words
   - **1.14+ range**: Basic structure learned but significant room for improvement
   - Lower loss consistently correlates with better text generation quality

5. **Device and Platform Impact:**
   - **CUDA GPU (Kaggle)**: Faster training, enabled 5,900-step experiments in reasonable time
   - **MPS (Apple Silicon)**: Capable of training but slower, suitable for smaller experiments
   - **CPU**: Feasible but very slow, recommended only for testing
   - Kaggle's free GPU resources were essential for conducting multiple experimental runs

6. **Sampling Method Comparison:**
   - **Top-K sampling (K=10)**: Restricts selection to the top 10 most likely characters, which can limit diversity but may improve coherence in smaller models
   - **Temperature sampling (0.7)**: Provides a good balance between creativity and coherence, working particularly well with larger, better-trained models
   - All high-quality outputs used temperature=0.7, suggesting it's a robust choice for this task

7. **Character-Level Learning Achievements:**
   - Models learned authentic Shakespeare character names without explicit supervision (QUEEN MARGARET, HASTINGS, BAPTISTA, PETRUCHIO)
   - Successfully captured character relationships and pairings from the same plays
   - Learned Early Modern English grammar patterns ("dost thou," "Is't," "'ld")
   - Demonstrated understanding of dramatic structure with proper dialogue turns
   - Best runs produced text nearly indistinguishable from actual Shakespeare at the character level

8. **Trade-offs and Practical Considerations:**
   - **Smaller models (10.79M)**: Train faster, require less GPU memory, produce lower quality text
   - **Larger models (85.20M)**: Require more computational resources, longer training time, but can generate excellent results
   - **Training multiple runs**: Recommended for important applications due to high variability in outcomes
   - **Resource availability**: Kaggle provides free GPU access, making experimentation accessible

**Overall Conclusion:**

These comprehensive experiments, conducted on Kaggle with CUDA GPU acceleration, demonstrate several important findings about character-level GPT models for Shakespeare text generation:

1. **Scaling Works**: Increasing model size from 10.79M to 85.20M parameters dramatically improves generation quality, validating the scaling hypothesis even at the character level.

2. **Training is Stochastic**: The same architecture and hyperparameters can produce vastly different results (loss ranging from 0.2446 to 0.6972), highlighting the importance of running multiple training experiments and selecting the best checkpoint.

3. **Quality Ceiling**: The best-performing model (Output 4, loss 0.2446) generated remarkably authentic Shakespeare-like text with proper character names, coherent dialogue, and minimal errors - demonstrating that character-level models can achieve near-human quality on constrained domains.

4. **Practical Accessibility**: Using free Kaggle GPU resources, these experiments show that high-quality language model research is accessible to students and researchers without expensive computational infrastructure.

5. **Architecture Validation**: The 12-layer, 768-dimensional GPT architecture successfully demonstrates the core Transformer concepts (self-attention, residual connections, layer normalization) and their effectiveness for language modeling.

The variability across training runs (Outputs 2-5 all using 85.20M parameters) underscores a crucial lesson: in deep learning, architecture and hyperparameters are only part of the story - training dynamics, initialization, and stochastic factors play equally important roles in achieving optimal performance.

---

## Model Summary

### Architecture Overview

| Component | Configuration |
|-----------|---------------|
| Model Type | Decoder-only Transformer (GPT) |
| Number of Layers | 12 |
| Embedding Dimension | 768 |
| Number of Attention Heads | 8 |
| Head Dimension | 96 (768 / 8) |
| Feedforward Dimension | 3072 (4 × 768) |
| Context Length (Block Size) | 128 characters |
| Vocabulary Size | 65 unique characters |
| Dropout Rate | 0.1 |
| Total Parameters | ~85.20 million |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Batch Size | 64 |
| Learning Rate | 3e-4 (0.0003) |
| Optimizer | AdamW |
| Max Iterations | 6000-8000+ (varied across experiments) |
| Device | GPU (Kaggle), MPS (Apple Silicon), or CPU |
| Loss Function | Cross-Entropy Loss |

### Dataset

| Attribute | Details |
|-----------|---------|
| Name | Tiny Shakespeare |
| Source | [char-rnn repository](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) |
| Size | ~1 MB of text |
| Content | Collection of Shakespeare's writings |
| Tokenization | Character-level |

---

## Installation and Requirements

### Prerequisites

This project requires the following Python packages:

- **PyTorch**: Deep learning framework for building and training the model
- **requests**: For downloading the Tiny Shakespeare dataset

### Installation Steps

1. **Install PyTorch**: Follow the instructions at [pytorch.org](https://pytorch.org/get-started/locally/) based on your system configuration (CPU, CUDA, or Apple Silicon MPS).

   For example, on a system with CUDA support:
   ```bash
   pip install torch torchvision torchaudio
   ```

   For Apple Silicon (M1/M2/M3):
   ```bash
   pip install torch torchvision torchaudio
   ```

2. **Install requests**:
   ```bash
   pip install requests
   ```

### Hardware Recommendations

- **Minimum**: CPU with at least 8GB RAM (training will be slow)
- **Recommended**: GPU with at least 4GB VRAM (NVIDIA CUDA or Apple MPS)
- **Optimal**: GPU with 8GB+ VRAM for faster training and larger batch sizes

**Note**: This project was developed using Kaggle notebooks which provide free GPU acceleration, making experimentation with different hyperparameters feasible.

---

## How to Run

### Training the Model

1. **Open the notebook**: Load `NewDP copy.ipynb` in Jupyter Notebook, JupyterLab, or Kaggle
2. **Run all cells**: Execute the cells sequentially from top to bottom
3. **Monitor training**: Watch the loss decrease over iterations (printed every 100 steps)
4. **Wait for completion**: Training takes 30-60 minutes on a modern GPU, several hours on CPU

The model will automatically:
- Download the Tiny Shakespeare dataset
- Initialize the model architecture
- Train for the specified number of iterations
- Display training progress and final loss

### Generating Text

After training completes, the notebook automatically runs inference with the prompt "O God, O God!" and generates 500 characters of Shakespeare-like text.

To generate text with a different prompt:

```python
# Change the context string to your desired prompt
context_str = "ROMEO:"  # or any other prompt
tokenized_context = tokenize(context_str)

# Generate text (adjust max_new_tokens for longer/shorter output)
y = model.generate(tokenized_context, max_new_tokens=500, temperature=0.7)

# Print the result
completion = tokens_to_string(y)
print(completion)
```

**Adjustable parameters:**
- `context_str`: The initial prompt (must contain only characters present in the training data)
- `max_new_tokens`: How many characters to generate (default: 500)
- `temperature`: Controls randomness (lower = more conservative, higher = more creative)

---

## References

This project is based on foundational research in Transformer architectures and language modeling:

1. **Vaswani, A., et al. (2017)**. "Attention Is All You Need." *Advances in Neural Information Processing Systems (NeurIPS)*.
   - Introduced the original Transformer architecture with multi-head attention

2. **Radford, A., et al. (2019)**. "Language Models are Unsupervised Multitask Learners." *OpenAI*.
   - Presented GPT-2, demonstrating the effectiveness of decoder-only Transformers for language modeling

3. **Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016)**. "Layer Normalization." *arXiv preprint arXiv:1607.06450*.
   - Introduced Layer Normalization, a key component for training deep Transformers

4. **Karpathy, A.** "char-rnn" and related educational materials.
   - Provided the Tiny Shakespeare dataset and inspiration for character-level language modeling

5. **Loshchilov, I., & Hutter, F. (2019)**. "Decoupled Weight Decay Regularization." *ICLR*.
   - Introduced AdamW optimizer used for training

---

## Acknowledgments

This project was developed as part of a course assignment to demonstrate practical understanding of modern language model concepts. Special thanks to:

- **Andrej Karpathy** for the Tiny Shakespeare dataset and educational resources on language modeling
- **The PyTorch team** for providing an excellent deep learning framework
- **Kaggle** for providing free GPU resources that enabled extensive hyperparameter experimentation
- **Course instructors** for guidance on Transformer architectures and deep learning best practices

---

## Project Contributors

- **ENGIN SAMET DEDE**
- **SEVGI DILAY DEMIRCI**

This project demonstrates the implementation of a character-level GPT language model from scratch, showcasing the core concepts of Transformer architectures, self-attention mechanisms, and autoregressive text generation.

