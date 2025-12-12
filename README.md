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

### Causal Self-Attention
In each block, the input data first undergoes Layer Normalization and then enters the Causal Self-Attention layer:

`x = x + self.attn(self.ln1(x))`
At this stage, each character looks at the characters that precede it in the sequence. When constructing the meaning of a character, the model learns which past characters are more important. The term causal means that the model cannot see future characters. This is a fundamental requirement of language modeling.
Thanks to the self-attention mechanism, the model can learn how, for example, the last letter of a word relates to the letters at the beginning of the word.
The residual connection (for example, the x + ... operation) ensures that the attention result is added to the input. This prevents information loss and allows deep architectures to be trained healthily.

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


