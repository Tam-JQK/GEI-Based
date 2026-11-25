"""Shared chronological splitting and history helpers."""
from datetime import datetime
import html
import re
import unicodedata
from zoneinfo import ZoneInfo
import numpy as np

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
