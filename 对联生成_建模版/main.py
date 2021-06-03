from preprocess_date import CoupletsDataset,load_dense_drop_repeat
from torch.utils.data import DataLoader
from utils import *
import os
from bilstm import Encoder, Decoder, Net, Attention
from torch.optim import Adam
from random import choice

#word2id, id2word, word_cnt = build_vocab()
train_path = './couplets/train'
test_path = './couplets/test'
model_path='./model/bilstm.pt'
get_emb_path='./get_emb/sgns.sikuquanshu.word'
emb_matrix, emb_vocab, size_emb_vocab, len_emb_vocab=load_dense_drop_repeat(get_emb_path)
word2id=emb_vocab['w2i']
id2word=emb_vocab['i2w']
emb_matrix=torch.tensor(emb_matrix)
word_cnt=len_emb_vocab
train_features = CoupletsDataset(word2id,train_path)
test_features=CoupletsDataset(word2id,test_path)
num_epochs = 10
batch_size=120
device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
emb_matrix=emb_matrix.to(device)
#test_features = CoupletsDataset(word2id, test_path)
train_dataloader = DataLoader(train_features, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True, )
#test_datalodaer=DataLoader(test_features, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True, )
print_num=100
# 初始化模型定义
en = Encoder(300, 128, word_cnt)
de = Decoder(word_cnt, 300, 128)
net = Net(en, de, device)
en = en.to(device)
de=de.to(device)
net=net.to(device)
loss_fn=nn.CrossEntropyLoss()
def train():
    optimizer = Adam(net.parameters(), lr=3e-5)
    net.zero_grad()
    num_step = 0
    best_loss=10
    average_loss=0
    for epoch in range(num_epochs):
        print("epoch:",epoch)
        net.zero_grad()
        for step, batch in enumerate(train_dataloader):
            net.train()
            input_ids, output_ids, lengths, mask, output_ids_flatten=batch
            """input_emb = []
            for i in range(len(input_ids)):
                for j in range(len(input_ids[0])):
                    input_emb.append(emb_matrix[(emb_vocab['w2i'][(emb_vocab['i2w'][input_ids[i][j]])])])
            input_emb = torch.tensor(input_emb)
            input_emb=input_emb.view(batch_size,-1,300)"""
            output_ids = torch.tensor(output_ids)
            mask = torch.tensor(mask)
            output_ids_flatten=torch.tensor(output_ids_flatten)
            input_ids=torch.tensor(input_ids)
            input_ids=input_ids.transpose(0,1)
            output_ids=output_ids.transpose(0,1)
            input_ids=input_ids.to(device)
            output_ids=output_ids.to(device)
            output_ids_flatten=output_ids_flatten.to(device)
            mask=mask.to(device)
            # print("上联",input_ids.size(),input_ids)
            # print("下联",output_ids.size(),output_ids)
            # print("output_ids_flatten",output_ids_flatten.size(),output_ids_flatten)
            # print("长度",lengths)
            # mask=mask.permute(1,0).contiguous()
            outputs=net(input_ids,emb_matrix, lengths, output_ids, output_ids_flatten, mask)
            outputs=outputs.permute(1,0,2).contiguous()
            abc=outputs.view(-1,outputs.shape[2])[mask.view(-1)==1.0]
            # print("abc",abc.size(),abc)
            # print("flatten",output_ids_flatten.size(),output_ids_flatten.view(-1))

            loss=loss_fn(abc,output_ids_flatten.view(-1))
            loss.backward()
            optimizer.step()
            if(loss<best_loss):
                best_loss=loss
            average_loss+=loss
            net.zero_grad()
            num_step += 1
            if num_step % print_num == 0 and num_step!=0:
                average_loss=average_loss/print_num
                print(num_step,"steps has been done,the loss has been:",loss,"the best loss is",best_loss,"average_loss=",average_loss)
                average_loss=0
                torch.save(net.state_dict(), model_path)


def generate_couplets():
    up_cpl='【'
    up_cpl+=input("请输入上联:")
    up_cpl = up_cpl.strip('\n')
    up_cpl+='】'
    down_cpl='下联:【'
    xx = []
    net.eval()
    net.zero_grad()
    for word in up_cpl:
        if word in word2id:
            xx.append(word2id[word])
        else:
            xx.append(word2id['【'])
  # xx为上联的id表示
    qq=xx
    up_len=len(xx)
    len1=up_len
    up_len=torch.tensor(up_len)
    up_len=up_len.unsqueeze(0)
    xx = torch.tensor(xx)
    xx=xx.unsqueeze(1)
    xx=xx.to(device)
    yy, atten_weight=net.predict(xx,emb_matrix,up_len,len1)
    yy=yy.squeeze(1)
    topv,topi=yy.topk(1)
    for i in  range(1,len1) :
        topa=topi[i][0]
        topa=topa.item()
        letter=id2word[topa]
        if qq[i]==4:
            letter=up_cpl[i]
        if  up_cpl[i-1]==up_cpl[i]:
            letter=down_cpl[i+2]
        down_cpl+=letter
    print(down_cpl)
def random_up():
    with open(os.path.join(test_path, 'in.txt'), "r", encoding='utf-8') as fr:
        raw=fr.readlines()
    up_cpl=choice(raw)
    while(len(up_cpl)>=25):
        up_cpl=choice(raw)
    xx = []
    up_cpl= ['【'] + up_cpl.strip().split(' ') + ['】']
    up="上联："
    for i in up_cpl:
        up+=i
    print(up)
    for word in up_cpl:
        if word in word2id:
            xx.append(word2id[word])
        else:
            xx.append(word2id['【'])
    qq = xx
    up_len = len(xx)
    len1 = up_len
    up_len = torch.tensor(up_len)
    up_len = up_len.unsqueeze(0)
    xx = torch.tensor(xx)
    xx = xx.unsqueeze(1)
    xx = xx.to(device)
    yy, atten_weight = net.predict(xx, emb_matrix, up_len, len1)
    yy = yy.squeeze(1)
    down_cpl = '下联:【'
    topv, topi = yy.topk(1)
    for i in range(1, len1):
        topa = topi[i][0]
        topa = topa.item()
        letter = id2word[topa]
        if qq[i] == 4:
            letter = up_cpl[i]
        if up_cpl[i - 1] == up_cpl[i]:
            letter = down_cpl[i + 1]
        down_cpl += letter
    print(down_cpl)



def main():
        te2tr=input("if continue train?")
        if(te2tr=='1'):
            print("train!")
            train()
        elif(te2tr=='2'):
            print("continue train")
            net.load_state_dict(torch.load(model_path))
            train()
        net.load_state_dict(torch.load(model_path))
        random_up()
        while(1):
            generate_couplets()

if __name__ == '__main__':
    main()