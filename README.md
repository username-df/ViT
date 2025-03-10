# ViT From Scratch

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/16d9122e-390e-4e1c-8614-8bfefb6e2d02" width="500" style="margin-right: 20px;" />
</div>

The transformer is usually used for natural language processing, but the Vision Transformer (ViT) is the application of the transformer encoder for computer vision tasks.

## Patch Embedding

The first step of the Vision Transformer is turning the image into patches and embedding them.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/816302ac-2800-478c-bcb5-42cbb03e58e9" width="300" style="margin-right: 20px;" />
</div>

<br></br>

Using the einops library, images can be reshaped from (batch_size, channels, height, width) to (batch_size, num_patches, dimensionality of each patch = 3 * patch_size * patch_size)

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/cad1f48d-44e6-4c34-99a7-17273bac6638" width=700" style="margin-right: 20px;" />
</div>

<br></br>

The patches are then lineary embedded by using a nn.Linear layer which takes each patch and outputs a embedding vector.

(batch_size, num_patches, dimensionality of each patch) * (dimensionality of each patch, embed_vector_size) -> (batch_size, num_patches, embed_vector_size)

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/2734da53-a8ad-4afc-ae7d-81a6cd64e51e" width=700 style="margin-right: 20px;" />
</div>

<br></br>

Positional encoding is important for the transformer since it doesn't look at sequences step by step like RNNs, it looks at the whole sequence in parallel, so positional information needs to be explicitly added.
In the original Transformer paper, this was done manually using sine and cosine functions; the Vision Transformer uses a learnable positional encoding.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/fb728463-657b-4bd9-b8ab-74649570249c" width=850 style="margin-right: 20px;" />
</div>
<br></br>

The CLS token is also used, similar to BERT. The CLS token summarizes the information obtained from the whole sequence.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/f1063545-4959-4aeb-8b09-94246eee7938" width=800 style="margin-right: 20px;" />
</div>

<br></br>

The CLS token is concatenated to the start of the embedded patch sequence and the positional encoding is added to get the final patch embedding.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/4024c3a0-d1eb-4bbf-ae31-194380e0ee02" width=400 style="margin-right: 20px;" />
</div>

## Multi-Head Attention

The patch embedding is then used as input to the Transformer Encoder. The main component of the Transformer Encoder is Multi-Head Attention.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/36c0d77b-3720-492e-a7f0-4558d8a89393" width=200 style="margin-right: 20px;" />
</div>

Self-Attention allows the Vision Transformer to focus on the relevant parts of the patch sequence by calculating a weighted sum of transformed patch embeddings. This prioritizes parts of the image containing important features like edges or textures, while reducing the impact of less important parts.

This done by taking the input sequence and transforming it, using learned linear layers, into three different matrices, the queries (Q), the keys (K), and the values (V). A dot product between the queries and keys is calculated and then scaled by the square root of the dimensionality of the keys. A softmax is then used to normalize the scores between 0 and 1. A weighted sum between the softmax scores and the value matrix is computed to see the importance of each patch; the CLS token summarizes the information from the weighted sum.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/a66dd65d-86b5-4b6a-b9ce-b6aee4173c64" width=400 style="margin-right: 20px;" />
</div>

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/65c6eac4-a7e8-4259-8b8c-b1fc1487339e" width=400 style="margin-right: 20px;" />
</div>

Multi-Head Attention is an extension of Self-Attention, where instead of doing Self-Attention once, it is done h times parallel.  The dimensionality of the model is split evenly across h heads, the dimensionality of each head is d_model / h. Each head performs self-attention independently, creating their own set of queries, keys and values. The output of all the heads are concatenated into a single matrix and passed through another linear layer to produce the final output.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/79221301-2869-4233-9cee-01577d011c04" width=650 style="margin-right: 20px;" />
</div>

## Transformer Encoder


The Transformer Encoder takes Multi-Head Attention and adds skip connections, layer norm and a multi-layer perceptron. 

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/5b02ec54-1f20-4eec-85f1-22a189ab8478" width=400 style="margin-right: 20px;" />
</div>

## Vision Transformer (ViT)

The Vision Transformer takes the final CLS token result as input to a MLP which acts as a classifier for the image.

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/89ee9b29-9954-4ece-bf66-ec3abfa2f49b" width=500 style="margin-right: 20px;" />
</div>
