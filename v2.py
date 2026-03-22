import torch
import torch.nn as nn
from torch.nn import functional as F
# from google.colab import files
# files.upload()

#hyperparameters 
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 16 # what is the maximum context length for predictions?
max_iters = 20
eval_interval = 5
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_inters = 200
# numebr of dimension i.e. a 4 dimension vector equals [0,0,0,0,0,0]
n_embd = 48
# number of heads for self attention must be divisible by number of dimensions
n_head = 6
n_layer = 1

#percentage of dropout
dropout = 0.2
# ------------


torch.manual_seed(1337)

# read data
with open('data.txt', 'r', encoding="utf-8") as f:
    text = f.read()

# take unique chars from data and add to list
chars = sorted(list(set(text)))
# length of possible characters
vocab_size = len(chars)

# string to integer & integer to string
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

#map stoi for encoding and itos for decoding
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# pass data.txt file to encoder
data = torch.tensor(encode(text), dtype=torch.long)

# split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% of the data
train_data = data[:n] #first tensory of 90%
val_data = data[n:] #second tensor of 10%

def get_batch(split):

    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    # ix short for index. assign random index integer to tensor rannging from 0 to data length minus block size. btach size refers to the number of indices so batch size 4 could equal tensor([706094, 516259, 150661,  91611]) 
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # iterates through ix and for each index, does a lookup in the data.txt file and grabs block size number of characters
    x = torch.stack([data[i:i+block_size] for i in ix])
    
    # does the same as x but offset by one so that model can guess and check target value compute accuracy and "learn". 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # store to gpu if available otherwise store to cpu memory
    x,y = x.to(device), y.to(device)

    # returns the x and y tensor blocks essentiall chunks of data from the data.txt file shape of tensor is blocksize by batchsize.
    return x, y

# tells PyTorch not to track gradients inside this function.
@torch.no_grad()
def estimate_loss():
    
    # assign empty dictionary
    out = {}

    # calls BigLanguageModel and uses the eval method from nn.Module
    model.eval()  # disable dropout etc. for accurate loss measurement
    
    for split in ['train', 'val']:

        # assign a tensor with int values 0. size of tensor equal to eval_inters i.e. eval_inters 4 = [0.,0.,0.,0.]
        losses = torch.zeros(eval_inters)

        #iterate from 0 to eval_inters
        for k in range(eval_inters):

            # call get batch for training and value and assign to x (train) and y (val)
            X, Y = get_batch(split)
            
            # pass batch to BigramLanguageModel instance whcih calls the forward method and returns the logits and loss and assined to the below variables (logits, loss)
            
            logits, loss = model(X,Y)
            
            # for every number in the iteration (0 to eval_inters) assign zero digit the loss value
            losses[k] = loss.item()

        # add to out dictionary the mean of losses for both train and val on every batch
        out[split] = losses.mean()
    
    #calls the BigramLanguageModel instance and calls the train method in nn.Module
    model.train() # switch back to training mode so optimizer updates work

    # return the out dictionary which holds the computed loss one for training and one for val
    return out

# the core of self attention

class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        # nn.Linear breaks a single vector into smaller dimensions for matrix computation.
        # For example, a 48-dimensional vector is split into 6 heads, each of 8 dimensions.
        # The weights of nn.Linear are randomly initialized and updated during training.
        # This process is done three times—once for each of Key (K), Query (Q), and Value (V),
        # so each head produces three separate matrices, each with its own learned weights.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # mask future interactions a i.e. 
        # 100000 → The          (can only see itself)
        # 110000 → The Cat      (Cat can see The + itself)
        # 111000 → The Cat Sat  (Sat can see The, Cat, Sat)
        # 111100 → The Cat Sat On
        # 111110 → The Cat Sat On The
        # 111111 → The Cat Sat On The Map
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # randomly sets a percentage of nerons to 0 during training forcing the network not to rely too much on a single neuron and spread the learning across all neurons.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # B = batch size, T = sequence length (block size), C = embedding dimension
        # x has shape (B, T, C) after adding token and position embeddings
        B,T,C = x.shape
        
        # Pass x through linear layers to create the Key (k) and Query (q) matrices
        # Each linear layer transforms the original embedding dimension (C) into head_size (C per head)
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        
        # q = (4,16,6) = 16 rows 6 cols
        # k = (4,16,6) = 16 row 6 cols
        # rows and cols must be the same for matrix multiplcation
        # so for key flip T (block size) with C (head dimension) becoming k = (B,C,T)
        # q = (4,16,6) * k = (4,6,16) returning a 16 x 16 matrix
        # when multiplying matrices number of columns in first matrix must equal number of rows in second matrix
        # so q column = C (6) * k Row = C(6)
        # outer dimension determins the shape of the matrix 16 x 16 T * T


        # compute attention scores ("affinities")
        # dimension step 1 q0[0] * k0[0] + dimension step 2 q0[1] * k0[1]
        # Dot product sums over feature dimensions (head_size)
        # matrix multiplication is computing the dot products at once
        # attention[i,j]
        # * C ** -0.5 scales down the dot product ti keep scores stabled. 
        wei = q @ k.transpose(-2,-1) * C**-0.5 #(B, T, C) @ (B, C, T) -> (B, T, T)
        
        # Takes the lower-triangular mask of the right size ([:T, :T])
        # Sets all 0 entries → -∞
        # infinity instead of 0 as softmax will compute weight of 0 to future tokens. -inf wil cause token to ignore completely resulting in casual attention.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        
        # softmax turns attention scores → attention weights/probabilities that sum to 1 per query token, respecting the mask.
        # Softmax is literally turning raw scores into percentages (probabilities) that sum to 1 across each row.
        wei = F.softmax(wei, dim=-1) #(B, T, T)
        
        # randomly zeroes some values in the tensor
        # dropout to prevent overfitting.
        # token level dropout to prevent heavy reliance on specific tokens during aggregation
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        # Multiply attention weights by V to aggregate context for each token
        v = self.value(x) #(B, T, C)
        out = wei @ v #(B, T, T) @ (B, T, C) -> (B, T, C)

        # returns value
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    # multi-head attention is essentially combining the different “attention subspaces” computed by each head back into one unified representation.

    def __init__(self, num_heads, head_size):
        super().__init__()
        # You create multiple independent attention heads in parallel
        # Each head has its own Q, K, V weights
        # Example: num_heads=6, head_size=8 → 6 heads of 8 dimensions each
        # Intuition: Each head learns different types of relationships in the sequence
        # create number of Heads() in this example (6) with dimension of head_size (8) 
        # or in other words repeat Head(8) 6 times
        # i.e. [Head(8),Head(8),Head(8),Head(8),Head(8),Head(8)]
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # After concatenating all heads (shape (B, T, num_heads*head_size)), you project back to original embedding size n_embd
        # This mixes information from all heads into a single embedding per token
        self.proj = nn.Linear(n_embd, n_embd)
        # Regularization on the output
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Each head produces (B, T, head_size)
        # Concatenate along feature dimension → (B, T, num_heads * head_size)
        # dim-1 means last dimension i.e. C. So you concatenatng n_emb.
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Linear projection mixes all heads back into n_embd → (B, T, n_embd)
        out = self.proj (out)
        # Randomly zero some elements for regularization
        # feature level dropout on the find output embedding
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # expand dimension by * 4 i.e. 48 to 192
            # y = x @ W.T + b
            # Takes the token embedding x and multiplies it by a learnable weight matrix.
            # This produces many new combinations of the original features.
            # At this point, the token is “experimenting” with different mixes of its own features.
            # Weights are learned during training via backprop, so over time the network figures out the most useful combinations.
            # Tokens are computing their own internal combinations of features in the FeedForward layer. Each token mixes its features using the learnable weights, creating a richer representation. This refined representation is then passed back into the next self-attention layer, where it can interact with other tokens via QKV.
            nn.Linear(n_embd, 4 * n_embd),
            # Removes negative values → keeps only active/meaningful combinations.
            # ntroduces non-linearity, which allows the network to represent complex functions (otherwise two linear layers would collapse into one).
            nn.ReLU(),
            # Merges the expanded features back into the original embedding size.
            # The output token embedding now contains richer information than the original.
            nn.Linear(4 * n_embd, n_embd),
            # Randomly zeroes some elements → prevents over-reliance on any single feature
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Block behaves as a mini-orchestra inside the transformer 
class Block(nn.Module):
    """ Transformer block: communincation followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we like
        super().__init__()
        head_size = n_embd // n_head # headsize = number of dimensions divided by number of heads. Each head gets a slice of the embedding vector.

        # Invokes instances of the SelfAttention, FeedForward and LayerNorm layers.
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        # initialize nn.Module (parent class)
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        # initialise matrix with random vectors. vocab_size (row) by n_embd (col). 
        # higher n_embd equals more dials for more expression when fine tuning. n_embd must be divisble by n_head.  
        # this must match vocab size to get enough rows to assign all posible tokens.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 

        # add position information to token to understand context i.e. dog bites man vs man bites dog. Without position model wont understand the sequence of the context.
        # must match the number of sequences a block can hold for context length i.e. block size of 16 means a matrix of 16 rows to assign each context with a 48 dimension vector.  
        self.position_embedding_table = nn.Embedding(block_size, n_embd) 

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    #idx = xb and target = yb called 
    def forward(self, idx, targets=None):
        # B = block size and T = batch Size initialised in getbatch.
        B,T = idx.shape
 
        #idx and targets are bot (B,T) tensor of integers
        # this is the “base meaning” of the token, looked up from the embedding table using the token’s index. Shape (B, T, C).
        # you are essentially assigning a token from x into a n_embd (48 for example) dimensional vector. 
        tok_emb = self.token_embedding_table(idx) # (B,T,C)

        # this is the positional encoding, giving the model a sense of where the token is in the sequence. Shape (T, C).
            # you are essentially assigning a token from y into a n_embd (48 for example) dimensional vector. This
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T,C)

        # combining the base meaning with positional context i.e. [0.5] + [0.01] = 0.51
        x = tok_emb + pos_emb

        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x)# (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,C)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on the train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
context = torch.zeros((1, 1), dtype=torch.long, device = device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

