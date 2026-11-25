"""Per-user splits shared by graph and sequence recommenders; ML-1M excluded."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, save_npz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'common/engine'))
from analysis.artifacts import file_hash,write_json
from evaluation_candidates import ensure_ranking_candidates

DATASETS = {'steam': 'publisher'}
REPORT=ROOT/'preprocessing/reports/per_user_v1'
OUT=ROOT/'processed_dataset'
EMPTY={'','none','nan','unknown','generic','unbranded','n/a','null','self-published','independently published'}


def now():return datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds')


def clean(value):
    if not isinstance(value,str):return None
    value=re.sub(r'\s+',' ',unicodedata.normalize('NFKC',html.unescape(value))).strip()
    value=re.sub(r'^by\s+','',value)
    value=re.sub(r'^Visit the (.+) Store$',r'\1',value,flags=re.I)
    if value.casefold() in EMPTY or not any(c.isalpha() for c in value):return None
    return value


def assignments(frame,random=False):
    frame=frame.sort_values(['u','t','i'],kind='stable').reset_index(drop=True)
    n=frame.groupby('u',sort=False).u.transform('size').to_numpy()
    position=frame.groupby('u',sort=False).cumcount().to_numpy()
    holdout=np.maximum(1,(n*.1).astype(np.int64))
    labels=np.where(position<n-2*holdout,0,np.where(position<n-holdout,1,2)).astype(np.int8)
    frame['s']=labels
    return frame


def filter_and_split(frame):
    records=[]
    while True:
        before=len(frame)
        for key in ['i','u']:
            count=frame[key].value_counts()
            frame=frame[frame[key].isin(count.index[count>=5])]
        sizes=frame[['i','p']].drop_duplicates().groupby('p').size()
        frame=frame[frame.p.isin(sizes.index[sizes>=5])]
        frame=assignments(frame)
        positives=frame.loc[frame.y==1].groupby(['u','s']).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
        # Keep short-history users: two training positives give at least one
        # real next-item example; short contexts use explicit PAD + mask.
        # Users whose held-out rows are all negative remain in the graph;
        # positive-only ranking evaluation naturally omits those users in that
        # split. Do not delete their useful training interactions.
        keep=positives.index[positives[0]>=2]
        frame=frame[frame.u.isin(keep)]
        record=dict(iteration=len(records)+1,before=before,after=len(frame),users=int(frame.u.nunique()),items=int(frame.i.nunique()))
        records.append(record);print(now(),'FILTER',record,flush=True)
        if not len(frame):raise ValueError('Empty data after warm-user and sequence-history filtering')
        if len(frame)==before:break
    return frame.reset_index(drop=True),records


def histories(frame):
    # Train histories are strictly earlier TRAIN positive events. Every held-out
    # target uses only the final five TRAIN positives, never held-out events.
    positive=((frame.y==1)&(frame.s==0)).to_numpy(np.int64)
    before=np.cumsum(positive)-positive
    starts=np.r_[0,np.flatnonzero(frame.u.to_numpy()[1:]!=frame.u.to_numpy()[:-1])+1]
    baseline=np.repeat(before[starts],np.diff(np.r_[starts,len(frame)]))
    lengths=before-baseline
    full=lengths>=5
    positions=np.flatnonzero(positive)
    result=[[] for _ in range(len(frame))]
    selected=np.flatnonzero(full)
    if len(selected):
        prior=positions[before[selected,None]-5+np.arange(5)]
        values=frame.i.to_numpy()[prior]
        for index,row in zip(selected,values):result[index]=row.tolist()
    item_values=frame.i.to_numpy()
    for index in np.flatnonzero(~full):
        count=int(lengths[index])
        if count:result[index]=item_values[positions[before[index]-count:before[index]]].tolist()
    return result,lengths>0


def prepare(dataset):
    target=OUT/dataset
    if target.exists():
        config=yaml.safe_load((target/'process_config.yaml').read_text())
        if config.get('split_protocol')!='per_user_warm_v1':raise ValueError('Refusing to overwrite another data version')
        return json.loads((target/'audit.json').read_text())
    field=DATASETS[dataset];bc=dataset=='Book-Crossing'
    uid='user_id:token';iid='ISBN:token' if bc else ('product_id:token' if dataset=='steam' else 'item_id:token')
    title='title:token_seq' if bc else 'title:token';rating='play_hours:float' if dataset=='steam' else 'rating:float'
    timestamp=None if bc else 'timestamp:float';threshold=5 if bc else (10 if dataset=='steam' else 3)
    raw=ROOT/'data'/'raw'/dataset
    metadata=pd.read_csv(raw/(dataset+'.item'),sep='\t',dtype=str).drop_duplicates(iid)
    values=metadata[field+':token'].map(clean) if field+':token' in metadata else pd.Series(None,index=metadata.index,dtype=object)
    recovery=ROOT/'dataset_metadata/recovered'/f'{dataset}.suppliers.tsv'
    if recovery.exists():
        recovered=pd.read_csv(recovery,sep='\t',dtype=str).set_index('item_id')['supplier']
        fallback=metadata[iid].map(recovered).map(clean)
        values=values.where(values.notna(),fallback)
    elif field in {'publisher','studio'} and dataset!='steam' and not bc:
        raise FileNotFoundError(f'Awaiting recovered {field}: {recovery}')
    metadata['_supplier']=values
    metadata=metadata.dropna(subset=[iid,title,'_supplier'])
    metadata=metadata[metadata[title].str.strip().ne('')]
    metadata['_key']=metadata['_supplier'].str.casefold()
    display=metadata.groupby('_key')['_supplier'].first().to_dict()
    allowed=set(metadata[iid]);parts=[];raw_rows=0
    cols=[uid,iid,rating]+([timestamp] if timestamp else [])
    for part in pd.read_csv(raw/(dataset+'.inter'),sep='\t',usecols=cols,dtype={uid:str,iid:str},chunksize=1_000_000):
        raw_rows+=len(part);part=part.dropna(subset=cols)
        if bc:part=part[part[rating]>=5]
        parts.append(part[part[iid].isin(allowed)])
    frame=pd.concat(parts,ignore_index=True).sample(frac=1,random_state=42).drop_duplicates([uid,iid])
    frame=frame.merge(metadata[[iid,'_key']],on=iid,validate='many_to_one')
    u,original_users=pd.factorize(frame[uid],sort=True);i,original_items=pd.factorize(frame[iid],sort=True)
    p,original_suppliers=pd.factorize(frame['_key'],sort=True)
    times=frame[timestamp].to_numpy(np.float64) if timestamp else np.random.default_rng(2025).random(len(frame))
    records=pd.DataFrame(dict(u=u.astype(np.int32),i=i.astype(np.int32),p=p.astype(np.int32),
        y=(frame[rating].to_numpy()>=threshold).astype(np.int8),t=times))
    matched_rows=len(records)
    del frame,parts
    records,iterations=filter_and_split(records)
    uc,kept_users=pd.factorize(records.u,sort=True);ic,kept_items=pd.factorize(records.i,sort=True)
    records['u']=uc.astype(np.int32);records['i']=ic.astype(np.int32)
    item_provider=np.full(len(kept_items),-1,np.int32);item_provider[records.i]=records.p
    sizes_before=np.bincount(item_provider,minlength=len(original_suppliers))
    grouped=np.where(sizes_before[item_provider]<10,-1,item_provider)
    supplier_keys,providers=np.unique(grouped,return_inverse=True)
    sizes=np.bincount(providers)
    context,eligible=histories(records)
    padding_id=len(kept_items)
    context_mask=[[0]*(5-len(v))+[1]*len(v) for v in context]
    context=[[padding_id]*(5-len(v))+v for v in context]
    group=field+':token'
    exported=pd.DataFrame({uid:records.u.to_numpy(),iid:records.i.to_numpy(),
        'history_behaviors':context,'history_mask':context_mask,'label:float':records.y.to_numpy(),group:providers[records.i],
        'sequence_eligible':eligible.astype(np.int8),'split':records.s.to_numpy()})
    if timestamp:exported[timestamp]=records.t.to_numpy()
    staging=OUT/(dataset+'.preparing');staging.mkdir(parents=True,exist_ok=False)
    for code,name in [(0,'train'),(1,'valid.CTR'),(2,'test.CTR')]:
        exported[exported['split']==code].drop(columns='split').to_csv(staging/(dataset+'.'+name),sep='\t',index=False)
    seq=exported[(exported['split']==0)&eligible&(exported['label:float']==1)].drop(columns='split')
    seq.to_csv(staging/(dataset+'.train.sequential'),sep='\t',index=False)
    config=dict(dataset=dataset,user_id=uid,item_id=iid,group_id=group,label_id='label:float',
        timestamp=timestamp,text_id=title,user_num=len(kept_users),item_num=len(kept_items),group_num=len(supplier_keys),
        candidate_mode='full',candidate_seed=2025,candidate_version=1,sample_num=200,
        exclusion='all_retained_positive_interactions_except_current_split_positives',
        split_protocol='per_user_warm_v1',split_strategy='per_user_random' if bc else 'per_user_temporal',
        valid_ratio=.1,test_ratio=.1,split_seed=2025,history_length=5,history_field='history_behaviors',
        history_strategy='train_only_strict_past_for_training_and_train_prefix_for_evaluation',
        train_file=dataset+'.train',sequential_train_file=dataset+'.train.sequential',
        min_train_positive=2,min_valid_positive=0,min_test_positive=0,
        history_padding_id=padding_id,sequence_item_embedding_num=padding_id+1,
        history_padding_policy='left_pad_outside_real_catalog_with_attention_mask',
        item_val=5,user_val=5,group_val=5,group_aggregation_threshold=10,
        label_threshold=threshold,min_rating=5 if bc else None,multi_provider=False,
        meta_group_num=3,meta_group_types=['size','freq'],processed_data_dir=str(target),reprocess=False)
    (staging/'process_config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True))
    for filename,values in [('iid2pid',{str(j):int(v) for j,v in enumerate(providers)}),
        ('uid2original',{str(j):str(original_users[v]) for j,v in enumerate(kept_users)}),
        ('iid2original',{str(j):str(original_items[v]) for j,v in enumerate(kept_items)}),
        ('pid2provider',{str(j):'Other' if v<0 else display[str(original_suppliers[v])] for j,v in enumerate(supplier_keys)})]:
        write_json(staging/(filename+'.json'),values)
    titles=metadata.set_index(iid)[title]
    write_json(staging/'iid2text.json',{str(j):str(titles.loc[original_items[v]]) for j,v in enumerate(kept_items)})
    save_npz(staging/'providers.npz',csr_matrix((np.ones(len(kept_items),np.int8),(np.arange(len(kept_items)),providers)),shape=(len(kept_items),len(supplier_keys))))
    train=exported[exported['split']==0];frequency=np.bincount(providers[train[iid]],minlength=len(supplier_keys))
    assignments={}
    for kind,values in [('size',sizes),('freq',frequency/sizes)]:
        groups=np.searchsorted(np.percentile(values,[100/3,200/3]),values,side='right');assignments[kind]=groups
        write_json(staging/f'pid2mgid_{kind}.json',{str(j):int(v) for j,v in enumerate(groups)})
    write_json(staging/'provider_stats.json',{str(j):dict(item_count=int(sizes[j]),freq=int(frequency[j]),avg_freq=float(frequency[j]/sizes[j]),meta_group_size=int(assignments['size'][j]),meta_group_freq=int(assignments['freq'][j])) for j in range(len(sizes))})
    positive_train=train[train['label:float']==1];seen_users=set(positive_train[uid]);seen_items=set(positive_train[iid])
    split_stats={}
    for code,name in [(0,'train'),(1,'valid'),(2,'test')]:
        subset=exported[(exported['split']==code)&(exported['label:float']==1)]
        split_stats[name]=dict(rows=int((exported['split']==code).sum()),positive_rows=len(subset),users=int(subset[uid].nunique()),cold_users=len(set(subset[uid])-seen_users),cold_items=len(set(subset[iid])-seen_items))
    assert split_stats['valid']['cold_users']==split_stats['test']['cold_users']==0
    assert set(seq[uid])==seen_users==set(range(len(kept_users)))
    assert all(len(v)==5 for v in exported.loc[exported['split']>0,'history_behaviors'])
    ensure_ranking_candidates(dataset,config,staging)
    audit=dict(dataset=dataset,created=now(),split_protocol='per_user_warm_v1',raw_rows=raw_rows,
        matched_deduplicated_rows=matched_rows,users=len(kept_users),items=len(kept_items),
        suppliers_before_merge=int(np.sum(sizes_before>0)),suppliers_after_merge=len(supplier_keys),
        all_interactions=len(records),sequence_training_rows=len(seq),splits=split_stats,iterations=iterations,
        raw_sha256={str((raw/(dataset+ext)).relative_to(ROOT)):file_hash(raw/(dataset+ext)) for ext in ['.inter','.item']},
        note='Per-user split BEFORE history construction. Full training prefix retained for graph models; sequence view uses positive training targets with at least one strictly earlier positive. PAD=item_num is outside the real candidate catalog; sequence models need item_num+1 embeddings and history_mask. Evaluation histories use training only. Cold items are reported, never silently removed.')
    write_json(staging/'audit.json',audit);staging.rename(target)
    print(now(),'COMPLETE',json.dumps(audit,ensure_ascii=False),flush=True)
    return audit

