"""Cross-Attention 훈련 / 그룹 CV / holdout 검증.

주의: 이미지를 64x64로 resize하며 grayscale 밝기 정보를 사용.
센서 시퀀스는 현재를 포함하여 동일 experiment_id의 과거 관측만 사용.
미래 데이터 사용 금지. 이미지/센서 정규화는 train fold에서만 fit.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import copy
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,confusion_matrix,classification_report
from sklearn.model_selection import GroupKFold
from organizer_features import M, REPO_ROOT
from models.cross_attention_model import CrossAttentionNet


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_frame(root, split):
    df=pd.read_csv(root/split/f'{split}.csv',dtype={'sample_id':'string','experiment_id':'string'})
    paths=M.resolve_image_paths(df,[root/split,root])
    if not df['sample_id'].is_unique: raise ValueError('Duplicate sample ID')
    return df,paths.tolist()


def make_sequences(df, seq_len=16):
    """Row count based sequence (not fixed seconds), latest item is current row."""
    if not df.relative_time_s.notna().all(): raise ValueError('Missing relative_time_s')
    sensors=M.numeric(df,M.SENSORS).to_numpy(dtype=np.float32)
    sequences=np.full((len(df),seq_len,len(M.SENSORS)),np.nan,dtype=np.float32)
    times=df.relative_time_s.to_numpy(dtype=float)
    for _,indices in df.groupby('experiment_id',sort=False).indices.items():
        ix=np.asarray(indices)
        ix=ix[np.argsort(times[ix],kind='mergesort')]
        for j,idx in enumerate(ix):
            history=ix[max(0,j-seq_len+1):j+1]
            # Repeat earliest observed row for left padding: no future rows used.
            padded=np.r_[np.repeat(history[0],seq_len-len(history)),history]
            sequences[idx]=sensors[padded]
    return sequences


class DatasetIR(Dataset):
    def __init__(self, seq, paths, labels=None):
        self.seq=seq.astype(np.float32)
        self.paths=paths
        self.y=None if labels is None else np.asarray(labels,dtype=np.int64)-1
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as im:
            a=np.asarray(im.convert('L').resize((64,64)),dtype=np.float32)/255.0
        x=torch.from_numpy(a[None,...]); s=torch.from_numpy(self.seq[i])
        if self.y is None: return x,s
        return x,s,int(self.y[i])


def fit_transform_sequences(train_seq, valid_seq):
    n,d=train_seq.shape[0]*train_seq.shape[1],train_seq.shape[2]
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    scaler=StandardScaler()
    tr_flat=imputer.fit_transform(train_seq.reshape(-1,d))
    scaler.fit(tr_flat)
    def transform(seq):
        shape=seq.shape
        return scaler.transform(imputer.transform(seq.reshape(-1,d))).reshape(shape).astype(np.float32)
    return transform(train_seq),transform(valid_seq),imputer,scaler


def make_loader(seq,paths,y,batch_size,shuffle):
    return DataLoader(DatasetIR(seq,paths,y),batch_size=batch_size,shuffle=shuffle,num_workers=0)


def train_one(train_seq,train_paths,y_train,val_seq,val_paths,y_val,epochs,batch,lr,device,seed):
    seed_all(seed)
    t,v,imp,sc=fit_transform_sequences(train_seq,val_seq)
    tr_loader=make_loader(t,train_paths,y_train,batch,True)
    va_loader=make_loader(v,val_paths,y_val,batch,False)
    model=CrossAttentionNet().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    criterion=nn.CrossEntropyLoss()
    best=-1.0; best_state=None; patience=5; bad=0
    for epoch in range(epochs):
        model.train(); loss_sum=0
        for img,seq,y in tr_loader:
            img,seq,y=img.to(device),seq.to(device),y.to(device)
            optimizer.zero_grad()
            loss=criterion(model(img,seq),y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();loss_sum+=float(loss.item())*len(y)
        pred=predict_loader(model,va_loader,device)
        score=f1_score(y_val,pred,labels=M.LABELS,average='macro',zero_division=0)
        print(f'epoch={epoch+1:02d} loss={loss_sum/len(y_train):.4f} val_macro_f1={score:.4f}',flush=True)
        if score>best+1e-5:
            best=score;best_state=copy.deepcopy(model.state_dict());bad=0
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(best_state)
    return model,imp,sc,best

@torch.no_grad()
def predict_loader(model,loader,device):
    model.eval(); out=[]
    for batch in loader:
        img,seq=batch[0].to(device),batch[1].to(device)
        out.extend((model(img,seq).argmax(dim=1)+1).cpu().numpy().tolist())
    return np.asarray(out,dtype=int)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir',type=Path,required=True)
    p.add_argument('--out_dir',type=Path,default=REPO_ROOT/'runs'/'doyu'/'cross_attention')
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--batch_size',type=int,default=64)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--seq_len',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args()
    if not 1<=a.seq_len<=64: raise ValueError('seq_len must be 1..64')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    seed_all(a.seed)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    tr,paths=load_frame(a.data_dir,'train'); va,val_paths=load_frame(a.data_dir,'validation')
    if not set(tr.experiment_id).isdisjoint(set(va.experiment_id)):
        raise ValueError('train/validation experiment overlap')
    y=tr.label.to_numpy(dtype=int); yv=va.label.to_numpy(dtype=int)
    trseq=make_sequences(tr,a.seq_len); vaseq=make_sequences(va,a.seq_len)
    groups=tr.experiment_id.to_numpy()
    folds=min(a.folds,len(np.unique(groups)))
    if folds<2: raise ValueError('Need >=2 groups')
    cv=[]; best_epochs=[]
    for fold,(ii,jj) in enumerate(GroupKFold(n_splits=folds).split(tr,y,groups),1):
        print(f'\n=== Cross attention fold {fold} ===',flush=True)
        m,imp,sc,score=train_one(trseq[ii],[paths[i] for i in ii],y[ii],
                                 trseq[jj],[paths[i] for i in jj],y[jj],
                                 a.epochs,a.batch_size,a.lr,device,a.seed+fold)
        cv.append(score)
    print('Cross-Attention CV Macro F1:',np.mean(cv),'+/-',np.std(cv))
    # Train-only CV serves model selection; holdout validation used once as final check.
    # Train full encoder for fixed epochs. This is NOT checkpoint selection on holdout labels.
    train_seq,validation_seq,imp,sc=fit_transform_sequences(trseq,vaseq)
    model=CrossAttentionNet().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-3)
    criterion=nn.CrossEntropyLoss()
    loader=make_loader(train_seq,paths,y,a.batch_size,True)
    for epoch in range(a.epochs):
        model.train()
        for img,seq,target in loader:
            img,seq,target=img.to(device),seq.to(device),target.to(device)
            optimizer.zero_grad();loss=criterion(model(img,seq),target)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
    pred=predict_loader(model,make_loader(validation_seq,val_paths,yv,a.batch_size,False),device)
    val_f1=f1_score(yv,pred,labels=M.LABELS,average='macro',zero_division=0)
    print('Holdout validation Macro F1:',val_f1)
    pd.DataFrame(confusion_matrix(yv,pred,labels=M.LABELS)).to_csv(a.out_dir/'attention_confusion_matrix.csv',index=False)
    pd.DataFrame(classification_report(yv,pred,labels=M.LABELS,zero_division=0,output_dict=True)).transpose().to_csv(a.out_dir/'attention_classification_report.csv')
    result=va[['sample_id','experiment_id','relative_time_s','image_path']].copy()
    result['actual']=yv;result['pred']=pred
    result.to_csv(a.out_dir/'attention_validation_predictions.csv',index=False)
    checkpoint={'state_dict':model.cpu().state_dict(),'imputer':imp,'scaler':sc,
                'seq_len':a.seq_len,'seed':a.seed,'architecture':'CrossAttentionNet'}
    torch.save(checkpoint,a.out_dir/'attention_model.pt')
    (a.out_dir/'scores.json').write_text(json.dumps({'cv_scores':cv,'cv_mean':float(np.mean(cv)),
              'cv_std':float(np.std(cv)),'holdout_macro_f1':float(val_f1)},indent=2),encoding='utf-8')
    print('Saved',a.out_dir/'attention_model.pt')

if __name__=='__main__':main()
