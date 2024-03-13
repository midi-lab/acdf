
import torch
import torch.nn as nn
import torch.nn.functional as F

#decay .95, .25, 1e-5
class VectorQuantizerEMA(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, groups=1, commitment_cost=0.25, decay=0.95, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__() 
       
        print('num codes', num_embeddings)

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs, reinit_code=None):
        # BCHW -> BHWC
        #inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        

        # Flatten 
        flat_input = inputs.view(-1, self._embedding_dim)
     
   
        # May be we can even use cosine distance.
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        #print('distances before', distances[0])

        # distances *= self._ema_cluster_size.unsqueeze(0).repeat(distances.shape[0], 1)
        distances *= torch.sqrt(self._ema_cluster_size.unsqueeze(0).repeat(distances.shape[0], 1))
        #distances *= torch.sqrt(torch.sqrt(self._ema_cluster_size.unsqueeze(0).repeat(distances.shape[0], 1)))
        #distances *= torch.exp((self._ema_cluster_size.unsqueeze(0).repeat(distances.shape[0], 1) / 20.0))


        #print('distances after', distances[0])

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        #  EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            

            #print('distances shape', distances.shape)
            #print('encodings shape', encodings.shape)
            #print('encodings', encodings.min(), encodings.max(), encodings.mean())
            #print('encodings sum', torch.sum(encodings, 0).shape)



            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
 
            #print('clusters shape', self._ema_cluster_size.shape)
            #print('clusters', self._ema_cluster_size.data.cpu().numpy().tolist())

            #raise Exception()
       
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
 
       
        # BHWC -> BCHW
        return quantized.contiguous(), loss, encoding_indices




