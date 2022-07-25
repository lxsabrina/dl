from cmath import cos
from this import d
# https://github.com/zhaogaofeng611/TextMatch
# tips: 使用dropout 效果反而不好  

class DSSM(nn.Module):
    def __init__(self, dropout = 0.2, device = 'cpu') -> None:
        super().__init__()
        # layers
        self.embed = nn.Embedding(7901, 100)
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(dropout)
        # activation function
        self.sigmoid = nn.Sigmoid()
        self.relu  = nn.ReLU()

    
    def forward(self, a, b):
        a = self.embed(a).sum(1)
        b = self.embed(b).sum(1)

        a = self.relu(self.fc1(a))
        a = self.relu(self.fc2(a))
        a = self.relu(self.fc3(a))
        

        b = self.relu(self.fc1(b))
        b = self.relu(self.fc2(b))
        b = self.relu(self.fc3(b))

        cosine = torch.cosine_similarity(a, b, dim=1 , eps = 1e-8)
        cosine = self.relu(cosine)
        cosine = torch.clamp(cosine,0, 1)
        return cosine
