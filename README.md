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

At this stage, the model still knows nothing meaningful; tokens has only been converted into numbers.

### Token Embedding + Position Embedding


This is the most important step where the model gets to learn raw character information and turn this into meaningful numeric representations that we shall utilize in future implementations. Transformer architectures do not accept inputs in the form of characters or numbers; thus, inputs are previously converted to high-dimensional vectors.

The contribution to the model in this project is a character IDs-basedtensor. Yet, these IDs are just indexes but they do not have any semantic meaning themselves. Thus, the first step is the Token Embedding process.

During the token embedding phase, the identity of all characters is translated into a dense n nembd = 768-dimensional vectors of the nn.Embedding layer. Through this process, the model will be able to generate a learnable representation of every character. The characters with similar contexts drift towards one another in the embedding space over time. The model is based on learning relationships between characters and this process.

This is applied in the code line below:

`tok_emb = self.wte(idx)`

 In this case, idx is a character ID tensor of size (B, T) and the result of this line is an embedding tensor of size (B, T, n embd).
Nevertheless, it is not enough to use token embedding only. Transformer architectures lack knowledge of order information. That is, although the model may have the same characters in different sequences it is not necessarily perceived so. This problem is solved by Position Embedding.

During position embedding, an additional embedding vector which depicts the location of each character in the sequence is produced. These vectors give the model the position of the character in the sentence. In such a way, the model does not only learn information of the type of which character, but also of the type, where this character is information.

This is executed in the code as below:

`pos = torch.arange(0, T, dtype=torch.long, device=device)`
`pos_emb = self.wpe(pos)`

In this case, pos is an array of position indices (0 to T-1) and wpe self is an array of N empirical vectors of length nembd.
Once, the token embedding and the position embedding processes have been performed, these two vectors are added as a single representation:

`x = self.drop (tokemb + pos_emb)`

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


Layer Normalization operates on a single token ans across the embedding (feature) dimension.

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

