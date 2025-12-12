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



