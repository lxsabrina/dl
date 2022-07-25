#参考资料：
# 1、https://www.cnblogs.com/techflow/p/14107099.html （代码实践）
# 2、https://haiping.vip/2020/01/02/Wide&Deep-Learning/ （论文阅读）
# 总结， sparse 特征应用在 wide，deep 两个端都会输入使用
# wide: 记忆性， 输入特征是  Categorical 特征 + Categorical的组合特征
# deep: 泛化性， 输入特征是  Categorical 特征(embedding)+ Continuos 特征(归一化)
# wide & deep 论文是用来解决ranking问题的

import torch
from torch import nn
class WideAndDeep(nn.Module):
    # site_categorical 24个取值； app_categorical 是 32个取值 
    def __init__(self, dense_dim = 13, site_categorical_dim = 24, app_categorical_dim = 32):
        super(WideAndDeep, self).__init__()
        # 线性部分 
        self.logistic = nn.Linear(19, 1, bias=True)
        # embedding部分
        self.site_emb = nn.Embedding(site_categorical_dim,6)    # app_categorical 部分进行embedding
        self.app_emb = nn.Embedding(app_categorical_dim, 6)     # app_categorical 部分进行embedding
        # 融合部分
        self.fusion_layer = nn.Linear(12,6)
    
    def forward(self, x):
        site = self.site_emb(x[:,-2].long())
        app = self.app_emb(x[:,-1].long())
        emb = self.fusion_layer(torch.cat((site,app), dim =1))
        return torch.sigmoid(self.logistic(torch.cat((emb, x[:,:-2]), dim =1))) #wide部分和deep部分进行 concat后 -> 线性变换->sigmoid 输出
