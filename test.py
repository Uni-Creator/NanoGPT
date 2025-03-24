#Import the neccessary modules
import torch
import torch.nn as nn
from torch.nn import functional as F
import PyPDF2

torch.manual_seed(1337)


block_size = 8
batch_size = 32
max_iter = 5000
eval_intervals = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32
# device = "cpu"
print(device)


def process_large_pdf(file_path):
    
    text = ""
    
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            
            text += page.extract_text() + "\n"
            
    return text

#Extract raw data from the pdf
rawData = process_large_pdf('Nature_of_Code.pdf')

#Tokenize the data
chars = sorted(list(set(rawData)))
vocab_size = len(chars)

# print("".join(chars), vocab_size, len(rawData))

#Encode and Decode functions
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])


#Encode the raw data and convert it to tensors
data = torch.tensor(encode(rawData), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])


#Split the data into trainig and test datasets
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

# print(len(test_data))

#Divide the training data into (32, 8): 32 batches of chunks of lenght 8
def get_batch(split):
    
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i : i + block_size]  for i in ix] )
    y = torch.stack([data[i+1 : i + block_size + 1] for i in ix] )
    x,y = x.to(device), y.to(device)
    return x, y

# xb, yb = get_batch("train")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


#Implement the BiGramLanguage Model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
            
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        
        #idx -> (B,T)
        
        for _ in range(max_new_tokens):
            
            logits, loss = self(idx) # prediction
            logits = logits[:, -1, :] # (B,C), last time step
            
            probs = F.softmax(logits, dim=-1) #(B,C), softmax activation
            
            sum_probs = probs.sum(dim=-1)
            print(sum_probs)
            
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1), take a sample
            
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1), append the sample to the running sequence
            
        return idx
    
model = BigramLanguageModel()
m = model.to(device)
# logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)

#create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iter):
    
    if iter % eval_intervals == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch("train")
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    

context = torch.zeros((1,1), dtype=torch.long, device = device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

