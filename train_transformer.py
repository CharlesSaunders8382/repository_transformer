import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads # '//' represents integer division

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"
        #linear layers
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size) 

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim) 

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape: (N,heads, query_len, key_len)
        # values shape: (N,value_len,heads,heads_dim)
        # after einsum (N, query_len, heads, head_dim) the flatten last two dimensions

        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        X = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(X)
        out = self.dropout(self.norm2(forward + X))
        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size,embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
            for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            out = layer(out,out,out,mask)

        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value,key,query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length= x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        X = self.dropout((self.word_embedding(x)+self.position_embedding(positions)))

        for layer in self.layers:
            X = layer(X, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(X)
        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0, device="cuda", max_length=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size,embed_size,num_layers,heads,device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size,embed_size,num_layers,heads,forward_expansion, dropout, device, max_length)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out



class TransformerTrainer:
    def __init__(self, model, train_data_loader, loss_function, optimizer, device):
        self.model = model
        self.train_data_loader = train_data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs, print_every=100):
        # Set the model to training mode
        self.model.train()

        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0.0
            for i, (x, trg) in enumerate(self.train_data_loader):
                # Move input sequences and target sequences to device
                x = x.to(self.device)
                trg = trg.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(x, trg[:, :-1])

                # Compute the loss
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                loss = self.loss_function(output, trg)

                # Backward pass
                loss.backward()

                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Update the parameters
                self.optimizer.step()

                # Accumulate the total loss
                total_loss += loss.item()

                # Print the loss for every few iterations
                if (i + 1) % print_every == 0:
                    avg_loss = total_loss / print_every
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(self.train_data_loader)}], Loss: {avg_loss:.4f}")
                    total_loss = 0.0
                print("Current loss:", loss.item())
        print(output)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data  # Assuming your data is a list or numpy array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]  # Get a single example from the dataset

        # Assuming your sample has features and labels, you can retrieve them directly
        features = sample[0]
        label = sample[1]

        # Perform any necessary preprocessing or transformations on the sample and label

        # Return the preprocessed sample and its corresponding label
        return features, label

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x=torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 3, 2, 4, 7, 6, 2]]).to(device)
    print("x shape:", x.shape)
    print("trg shape:", trg.shape)

    data = [(x[i], trg[i]) for i in range(x.size(0))]

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    # Define your model
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

    # Define your optimizer
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define your data loaders, loss function, and other components for training

    train_dataset = MyDataset(data)  # Replace 'data' with your actual training data
    batch_size = 32
    num_epochs = 10

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loss_function = nn.CrossEntropyLoss()

    # Create an instance of TransformerTrainer
    trainer = TransformerTrainer(model, train_data_loader, loss_function, optimizer, device)

    # Call the train method to start the training process
    trainer.train(num_epochs=num_epochs, print_every=100)
    print("hello")