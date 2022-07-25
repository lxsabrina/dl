# conghuang的机器学习手记 https://zhuanlan.zhihu.com/p/332786045 
# deepFM

from socket import VM_SOCKETS_INVALID_VERSION
import torch


class FM(nn.Module):
    def __init__(self):
        super(FM, self).__init__()
       
    def forward(self, fm_input):
        square_of_sum = torch.pow(torch.sum(fm_input, dim=1),2) # [bs, emb_size]
        sum_of_square = torch.sum(fm_input * fm_input, dim =1) # [ bs, emb_size]
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term

class DeepFM(nn.Moduel):
    def __init__(self, cate_fea_nuniqs, nume_fea_size = 0 , emb_size = 8, hid_dims = [256, 128], num_classes = 1, dropout = [0.2, 0.2]):
        super(DeepFM, self).__init__()
        self.cate_fea_size = len(cate_fea_nuniqs)
        self.nume_fea_size = nume_fea_size

        '''FM '''
        # 一阶
        if self.nume_fea_size ! =0 :
            self.fm_1st_order_dense =  nn.Linear(self.nume_fea_size, 1)  #数值特征的一阶表示
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in cate_fea_numiqs])  #类别特征的一阶表示
        
        # 二阶
        self.fm_2nd_order_sparse_emb = nn.ModuleList([nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_numiqs]) #类别特征的二阶表示


        """DNN部分"""
        self.all_dims = [self.cate_fea_size * emb_size] + hid_dims
        self.dense_linear = nn.Linear(self.nume_fea_size, self.cate_fea_size * emb_size)  # 数值特征的维度变换到FM输出维度一致
        self.relu = nn.ReLU()
        # for DNN 
        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_'+str(i), nn.Linear(self.all_dims[i-1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_'+str(i), nn.Dropout(dropout[i-1]))
        # for output 
        self.dnn_linear = nn.Linear(hid_dims[-1], num_classes)
        self.sigmoid = nn.Sigmoid()

def train_and_eval(model, train_loader, valid_loader, epochs, device):
    best_auc = 0.0
    for _ in range(epochs):
        """训练部分"""
        model.train()
        print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        write_log('Epoch: {}'.format(_ + 1))
        train_loss_sum = 0.0
        start_time = time.time()
        for idx, x in enumerate(train_loader):
            cate_fea, nume_fea, label = x[0], x[1], x[2]
            cate_fea, nume_fea, label = cate_fea.to(device), nume_fea.to(device), label.float().to(device)
            pred = model(cate_fea, nume_fea).view(-1)
            loss = loss_fcn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.cpu().item()
            if (idx+1) % 50 == 0 or (idx + 1) == len(train_loader):
                write_log("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                          _+1, idx+1, len(train_loader), train_loss_sum/(idx+1), time.time() - start_time))
        scheduler.step()
        """推断部分"""
        model.eval()
        with torch.no_grad():
            valid_labels, valid_preds = [], []
            for idx, x in tqdm(enumerate(valid_loader)):
                cate_fea, nume_fea, label = x[0], x[1], x[2]
                cate_fea, nume_fea = cate_fea.to(device), nume_fea.to(device)
                pred = model(cate_fea, nume_fea).reshape(-1).data.cpu().numpy().tolist()
                valid_preds.extend(pred)
                valid_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model.state_dict(), "data/deepfm_best.pth")
        write_log('Current AUC: %.6f, Best AUC: %.6f\n' % (cur_auc, best_auc))

if __name__ == '__main__':
    
    data = pd.read_csv("criteo_sample_50w.csv")

    dense_features = [f for f in data.columns.tolist() if f[0] == "I"]
    sparse_features = [f for f in data.columns.tolist() if f[0] == "C"]

    data[sparse_features] = data[sparse_features].fillna('-10086', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    ## 类别特征labelencoder
    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    ## 数值特征标准化
    for feat in tqdm(dense_features):
        mean = data[feat].mean()
        std = data[feat].std()
        data[feat] = (data[feat] - mean) / (std + 1e-12)   # 防止除零

    print(data.shape)
    data.head()

    train, valid = train_test_split(data, test_size=0.2, random_state=2020)

    print(train.shape, valid.shape)

    train_dataset = Data.TensorDataset(torch.LongTensor(train[sparse_features].values), 
                                    torch.FloatTensor(train[dense_features].values),
                                    torch.FloatTensor(train['label'].values),)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=2048, shuffle=True)

    valid_dataset = Data.TensorDataset(torch.LongTensor(valid[sparse_features].values), 
                                    torch.FloatTensor(valid[dense_features].values),
                                    torch.FloatTensor(valid['label'].values),)
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=4096, shuffle=False)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    cate_fea_nuniqs = [data[f].nunique() for f in sparse_features]
    model = DeepFM(cate_fea_nuniqs, nume_fea_size=len(dense_features))
    model.to(device)
    loss_fcn = nn.BCELoss()  # Loss函数
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    # 打印模型参数
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print(get_parameter_number(model))

    # 定义日志（data文件夹下，同级目录新建一个data文件夹）
    def write_log(w):
        file_name = 'data/' + datetime.date.today().strftime('%m%d')+"_{}.log".format("deepfm")
        t0 = datetime.datetime.now().strftime('%H:%M:%S')
        info = "{} : {}".format(t0, w)
        print(info)
        with open(file_name, 'a') as f: 
            f.write(info + '\n') 
    
    #模型训练
    train_and_eval(model, train_loader, valid_loader, 30, device)
