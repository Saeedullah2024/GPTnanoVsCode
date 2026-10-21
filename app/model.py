import torch
import torch.nn as nn
import math

#First Step of Transformer --------> Input Words into Embedding layer into Vectors
class InputEmbeddings(nn.Module):
    
    def __init__(self , d_model : int , vocab_size : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size , d_model)
    
    def forward(self , x):
        return self.embedding(x) * math.sqrt(self.d_model)

#Secound Step of Transformer --------> Positional Encoding to add positional information to the input embeddings
class PositionalEncoding(nn.Module):
    
    def __init__(self , d_model : int , seq_len : int , dropout : float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len #To set the maximum length of the inpur sequence
        self.dropout = nn.Dropout(dropout) #To set at the time of training to avoid overfitting
        
        #Create a matrix of shape (seqlen and d model) (Row, Column)
        pe = torch.zeros(self.seq_len , d_model)
        #Create a vector of shape (seq len)
        position = torch.arange(0 , seq_len , dtype=torch.float).unsqueeze(1)  # -> this create 1-D (Just Rows One column) (row , )
        #Create the Divide term for the PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0 , d_model , 2).float() * (-math.log(10000.0) / d_model)) 
        #create for the Sin term for even index
        pe[: , 0::2] = torch.sin(position * div_term) #For the Even index
        pe[: , 1::2] = torch.cos(position * div_term) #Foe the Odd index
        pe = pe.unsqueeze(0) # (1 , seq_len , d_model)
        self.register_buffer('pe' , pe) #This will save the tensors in the model
    
    def forward(self , x):
        x = x + (self.pe[: , : x.shape[1] , : ]).requires_grad_(False)#tensor[dim0, dim1, dim2] and requiresgrad -> don't backpropagate through
        return self.dropout(x)
        