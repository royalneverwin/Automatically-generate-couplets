from preprocess_date import load_dense_drop_repeat
from torch.utils.data import DataLoader
import os
from bilstm import Encoder, Decoder, Net, Attention
from torch.optim import Adam
from random import choice
import torch
import torch.nn as nn

# flask init
from flask import Flask, session, render_template, request, redirect, send_from_directory
app = Flask(__name__)

#word2id, id2word, word_cnt = build_vocab()
test_path = './couplets/test'
model_path='./model/bilstm.pt'
get_emb_path='./get_emb/sgns.sikuquanshu.word'
emb_matrix, emb_vocab, size_emb_vocab, len_emb_vocab=load_dense_drop_repeat(get_emb_path)
word2id=emb_vocab['w2i']
id2word=emb_vocab['i2w']
emb_matrix=torch.tensor(emb_matrix)
word_cnt=len_emb_vocab
device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
emb_matrix=emb_matrix.to(device)
#test_features = CoupletsDataset(word2id, test_path)
#test_datalodaer=DataLoader(test_features, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True, )
en = Encoder(300, 128, word_cnt)
de = Decoder(word_cnt, 300, 128)
net = Net(en, de, device)
en = en.to(device)
de=de.to(device)
net=net.to(device)
loss_fn=nn.CrossEntropyLoss()

@app.route("/generate_cpl", methods=["POST", "GET"])
def generate_couplets_web():
    up_cpl = request.form["up_cpl"]
    up_cpl = '【' + up_cpl + '】'
    print(f"[INFO] up_cpl={up_cpl}")
    down_cpl = ''
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
    print(f"[INFO] down_cpl={down_cpl}")
    return down_cpl[:-1]

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

@app.route("/random_up", methods=["POST", "GET"])
def random_up_web():
    net.eval()
    net.zero_grad()
    with open(os.path.join(test_path, 'in.txt'), "r", encoding='utf-8') as fr:
        raw=fr.readlines()
    up_cpl=choice(raw)
    while(len(up_cpl)>=25):
        up_cpl=choice(raw)
    xx = []
    # up_cpl= up_cpl.strip().split(' ')
    up_cpl= ['【'] + up_cpl.strip().split(' ') + ['】']
    up="上联："
    for i in up_cpl:
        up+=i
    # print(up)
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
    down_cpl = ''
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
    # print(down_cpl)
    down_cpl = down_cpl[:-1]
    up_cpl = ''.join(up_cpl[1:-1])
    print(f"[INFO] up_cpl={up_cpl}")
    print(f"[INFO] down_cpl={down_cpl}")
    return {'up_cpl': up_cpl, 'down_cpl': down_cpl}

def random_up():
    net.eval()
    net.zero_grad()
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

@app.route("/")
def index() :
    return render_template("index.html")

def main():
    # net.load_state_dict(torch.load(model_path))
    random_up()
    while(1):
        generate_couplets()

if __name__ == '__main__':
    # net.load_state_dict(torch.load(model_path))
    random_up()
    app.run(port=80, debug=True)
    # main()