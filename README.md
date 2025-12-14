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
Causal Self-Attention: An in-depth Excursion.

The working sense beneath the Causal Self-Attention mechanism is described in our GPT implementation by relating the corresponding mathematical equations with the code.

---

## Overview

In each Transformer Block, the primary step is to standardize the input with Layer Normalization, and subsequently to put it through the layer called Causal Self-Attention. This whole process is summarized in one line of our code:
```python
x = x + self.attn(self.ln1(x))
```

What this line really does is to break down:

1. Existing representation x is normalized through self.ln1 first to ensure that the attention mechanism has a more stable input.
2. The normalized input is then subjected to self.attn(...) which collects some context information of the preceding positions in the sequence.
3. This computed new information is finally again added to the original representation using a residual connection to avoid the loss of information as data is passed through the network.

---

The first step is to get to know the shape of the input.

The token and positional embeddings are concatenated in the tensor entering this line, i.e. the value of x. Its shape is:

$$x \in \mathbb{R}^{B \times T \times C}$$

Where:

- **B** denotes the batch size (number of sequences that we process at a time)
- **T** sequence length (at most block size) (here, 128, i.e. block size of 128)
where - is embedding dimension (n embd 768 )

Practically speaking, we are dealing with a sequence of tokens (characters) and, thus, a 768-dimensional vector represents both the token itself and its location in the sequence.

---

The Role of LayerNorm (Pre-LN Design): Step 2.

The input is first channeled through the Layer Normalization before the process of attention computation starts:
```python
self.ln1 = nn.LayerNorm(n_embd)
# ...
self.ln1(x)
```

The LayerNorm acts by normalizing the mean and the variance of the embedding vectors of each token separately. To compute one token vector x b, t,: The calculation is:

$\text{LN} (x b, t,: ) = - gamma x odot (x b,t,: - mu)/ sq root of sigma 2 + epsilon + b.

In this case, the calculation of mu and s 2 is performed in the feature (embedding) dimension of that particular token. Learnable parameters of the model are the parameters of the model, which can be used to tune the normalized output as required by changing the parameters of the model to the values of the parameters.

The important point is that this normalization does not occur after the operation of attention, but it occurs before. This is an architectural design, which is referred to as the Pre-LN Transformer design, which offers greater training stability than the initial Post-LN approach. It specifies that the values fed into the attention mechanism are of the same scale ensuring that the training process does not have instabilities in terms of numerical values.

---

This step involves the generation of Query, Key and Value vectors.

After the normalized tensor has been fed into the attention layer, the first task will be generating three projections of each token Query (Q), Key (K), and Value (V). In our code:
```python
qkv = self.c_attn(x)                 # Linear: (B, T, C) + (B, T, 3C)
q, k, v = qkv.split(self.n_embd, dim=2)
```

This is done efficiently as a single linear transformation which gives out a three times larger embedding as a tenor which is further divided into three equal parts. This would be mathematically represented as:

$$[Q \; K \; V] = XW + b$$

After splitting:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

In which the LayerNorm output is denoted by $X$, and the learned weight matrices are denoted by $W_Q, W_K, W_V, and are conceptually distinct, but are grouped together in self.c_attn due to computational efficiency.

What are the intuitive meaning of Q, K, V?

- **Query (Q)**: Refers to what information am I seeking? for each token
- **Key (K)**: Refers to what I know. for each token
- **Value (V)** This is what information will I bring in case I am picked? for each token

The attention mechanism operates by matching queries with keys to come up with relevance, after which the values are weighted by the relevance scores.

---

Multi-Head Decomposition is the fourth step in the process.

In our model, 8 attention heads are used (n koning 8), so Q, K and V are decomposed into 8 subspaces. The code achieves this by operations of reshape and transpose:
```python
k = k.view(B, T, self.n_head, self.head-size)transpose(1, 2).
q = q.view(B, T, self.n head,self.head size).transpose(1, 2)
v = v.view(B, T, self.n_head, self.head size).transpose(1 2).
```

Following this change the shapes are:

Q, K, V are matrices, in the form of B h T dh, in Rd.

Where:

- $h = \text{n\_head} = 8$
- $d_h = C / h = 768 / 8 = 96$ (the dimension per head)

**Mutilple heads:** All heads can be taught to pay attention to various forms of relationships in the data. In language modeling, as an example, one head could be concerned with syntactic relationships (subject-verb agreement) and the other with semantic relationship (word meanings in context) and another head concerned with local character pattern. With 8 parallel attention computations the results of which are combined by the model, it has a far more useful insight into the input than a single attention mechanism would offer.

---

Calculating Attention Scores (Scaled Dot-Product) Step 5: Calculating Attention Scores (Scaled Dot-Product).

This is followed by the meat of the attention mechanism computing how much each token should pay attention to each other token. This is achieved by a dot product that is scaled:
```python
att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
```

Mathematically:

$$A = \frac{QK^T}{\sqrt{d_h}}$$

Multiplication of matrices q.transpose (-2, -1) compares the dot product of each query with each key and yields a T T matrix of uncoded attention scores per head and batch element.

**Why we divide by $\sqrt{d_h}$?** This is an important scaling factor that is essential to training stability. When it is absent, with a large head dimension, the dot products may become very large in magnitude. High values have the result of generating very peaked distributions by the ensuing softmax operation (essentially one-hot vectors), and in the process, vanishing gradients. Our scaling down of the network by a factor of the square root of d h makes the dot products fall in a range where softmax is trained more smoothly and effectively.

---

The sixth step is to apply the Causal Mask.

In language modeling, the main limitation is that the model cannot see future characters when it is predicting the next character. Through causal mask this is enforced:
```python
att = att.masked fill(att.bias[:,:,:T,:T] = 0, float( - inf ))
```

The mask is produced in the form of initialisation:
```python
self.register_buffer("bias" torch.tril(torch.ones(block_size, block-size))
                             .view(1, 1, block_size, block_size))
```

This forms a lower-triangular matrix of ones. We then mask the fill with the masked fill operation, which then fills all the positions where the mask has a value of zero (i.e.: future positions) to negative infinity.

The masked attention scores are mathematically adjusted to:

$A i j = - 1.5 = - 1 = - 2.5 = - 4 = - 5.5 = - 5 = - 6.5 = - 7.5 = - 8.5 = - 9.5 = - 10.5 = - 11.5 = - 12.5 = - 13.5 = - 14.5 = - 15.5 = - 16.5 = - 17.5 = - 18.5 = - 19.5

At position i, only position j 1 = i (the current and all the earlier positions) are visible. Setting the future positions to -inf is a way of making sure that they are set to be precisely equal to zero after the softmax operation, and hence they would be invisible to the model.

**Example: In the case, where the model is working on the 5th character of a sequence, it can only be able to attend to character 1, 2, 3, 4, and 5. Characters 6, 7, 8, etc. are totally out of view, as though they were not there.

---

## Step 7: Softmax Normalization

The attention matrix is masked, and then each row goes through a softmax:
```python
att = F.softmax(att, dim=-1)
```

Mathematically:

$\alpha i, j = e Ai, j)/ e Ai, k Sum k iLess.

This converts the uncoded scores into a probability distribution. The resultant values of alpha i,j will be the answer to the question: at what degree of attention should we pay to token i with the processing of token i?

The weights of attention of each position being the same result, as we divide by the sum of exponentials, over valid positions only (i.e. position number k at most). The causal mask values of $-\infty are made equal to 0 by softmax and do not add to the weighted sum.

---

The next step involves calculating the weight value of values.

Now that the weightes of attention have been obtained we can compute information by taking a weighted sum of the value vectors:
```python
y = att @ v  # (B, n_head, T, head_size)
```

Mathematically, in each position, i:

$$y_i = \sum_{j \leq i} \alpha_{i,j} V_j$$

Such is the essence of what attention is doing. A weighted sum of all tokens which such a token can observe, the weights indicating the learned relevance, is a new representation of this token. A token that is very relevant to the current position makes a larger contribution to the output; irrelevant tokens make lesser contributions.

The representation of each token has been filled with historical information about its background after this operation. The model has successfully been trained to look back and accumulate the concerned information.

---

He 9: Reassembling Heads and Output Projection.

The results of all 8 attention heads must be re-integrated into one representation:
```python
y = y.contiguous(1, 2).transpose(B, T, C).
return self.resid_dropout(self.c_proj(y))
```

This is used to concatenate the head outputs and a final linear projection is used. Mathematically:

$Y = W O Concatenate(head 1, head 2, head h)

Output projection W O enables the model to learn how to optimally pool the information obtained by the various heads. The dropout used in this case (resid dropout) is used to avoid overfitting, where a few elements are randomly zeroed.

---

The connection that remains is the one between Barbie and Ken.

Going back to the complete Block-level view:
```python
x = x + self.attn(self.ln1(x))
```

The residual (skip) connection combines the attention output with the input. Mathematically:

$$H' = H + \text{Attn}(\text{LN}(H))$$

It is a very minor addition, but it is essential to the success of deep transformer models. The residual connection is used in two significant functions:

1. **Information Preservation**: The attention mechanism does not erase an already existing contextual information, instead, it adds some contextual information. The initial token identity and positional data is still available to subsequent layers.

2. **Gradient Flow: In backpropagation**, it is possible to have gradients flowing directly through the addition operation, and never compute the attention at all. This forms a gradient highway that enables training signals to instead travel to early layers, rather than vanishing, which is required to train networks as deep as our 12-layer one.

It would be practically impossible to train deep transformers without residual connections. The gradients would fade away (becoming too small to cause any learning) or grow to extremely large values (becoming so large as to destabilize training) before reaching the initial layers.

---
To Sum up,

The representation of every token is converted by the Causal Self-Attention mechanism, which enables the representation of every token to collect relevant information based on all past positions of the sequence. The model learns by interaction between queries, keys and values across numerous heads which past tokens are needed to predict the next one.

This is necessary to avoid the autoregressive language modeling model cheating: the causal mask makes sure that the model will never peep at the future tokens. Together with Layer normalization to achieve stability and residual connections to achieve trainability, this mechanism is the basis of our GPT being able to generate coherent, Shakespeare-like text.

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

The data coming out of the Attention layer is passed through Layer Normalization again and then enters the Feed Forward Network (MLP) layer:
```python
x = x + self.mlp(self.ln2(x))
```

The MLP processes each token independently and applies non-linear transformations. This layer enables the contextual information gathered by attention to become a deeper and more abstract representation. Here, the dimension is first expanded (n_embd → 4 * n_embd), then reduced again. This structure is standard in Transformer architectures.

Again, the MLP output is added to the input using a residual connection, thus preserving information.

---

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

