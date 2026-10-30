#!/usr/bin/env python3
"""Decode each released AVI once; share lossless uint8 crops across models.

The cache is private derived data. It stores post-skip RGB crops at 36, 72,
and 128 pixels, unmodified BVP and session-level normalization statistics.
Only implementation code and aggregate results may enter the code release.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np

SIZES=(36,72,128)
SKIP=180

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while b:=f.read(1024*1024): h.update(b)
    return h.hexdigest()

def crop_box(boxes, width, height):
    valid=np.isfinite(boxes).all(axis=1)&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
    if not valid.any(): raise ValueError('No valid released bounding box')
    x1,y1,x2,y2=np.median(boxes[valid],axis=0).astype(int)
    w,h=x2-x1,y2-y1
    cx,cy=(x1+x2)//2,(y1+y2)//2
    nw,nh=int(w*1.5),int(h*1.5)
    out=[max(0,cx-nw//2),max(0,cy-nh//2),min(width,cx+nw//2),min(height,cy+nh//2)]
    if out[2]<=out[0] or out[3]<=out[1]: raise ValueError('ROI is outside video')
    return [int(v) for v in out],int(valid.sum())

def session_stats(arr):
    """Bounded-memory population moments; equivalent formula to full-session normalization."""
    count=0; total=0.; squares=0.; diff_count=0; diff_total=0.; diff_squares=0.
    for start in range(0,len(arr),64):
        block=np.asarray(arr[start:start+64],dtype=np.float32)
        count+=block.size
        total+=np.sum(block,dtype=np.float64)
        squares+=np.einsum('ijkl,ijkl->',block,block,dtype=np.float64,optimize=False)
        end=min(start+64,len(arr)-1)
        if end>start:
            a=np.asarray(arr[start:end],dtype=np.float32)
            b=np.asarray(arr[start+1:end+1],dtype=np.float32)
            diff=(b-a)/(b+a+np.float32(1e-7))
            diff_count+=diff.size
            diff_total+=np.sum(diff,dtype=np.float64)
            diff_squares+=np.einsum('ijkl,ijkl->',diff,diff,dtype=np.float64,optimize=False)
    mean=total/count
    std=float(np.sqrt(max(0.,squares/count-mean*mean)))
    dmean=diff_total/diff_count
    dstd=float(np.sqrt(max(0.,diff_squares/diff_count-dmean*dmean)))
    if std<=0 or dstd<=0: raise ValueError('Degenerate image normalization statistics')
    return {'mean':float(mean),'std':std,'diff_std':dstd}

def build_one(job):
    root,out,row,provenance=job
    started=time.time()
    cv2.setNumThreads(1)
    key=row['session']; source=Path(root)/row['participant_key']/row['task']
    dest=Path(out)/key
    complete=dest/'stats.json'
    if complete.exists():
        stats=json.loads(complete.read_text())
        if stats.get('schema_version')=='compact-v1' and stats['frames']==int(row['available_frames_after_skip']) and stats.get('cache_provenance')==provenance:
            return {'session':key,'status':'reused','frames':stats['frames'],'seconds':0}
        raise RuntimeError(f'Existing cache has incompatible provenance: {dest}')
    dest.mkdir(parents=True,exist_ok=True)
    bvp=np.genfromtxt(source/'BVP_synchronized.csv',delimiter=',',names=True)['bvp']
    boxes=np.genfromtxt(source/'face_bboxes.csv',delimiter=',',skip_header=1,ndmin=2)
    cap=cv2.VideoCapture(str(source/'video_bfdt_ffv1.avi'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    if not cap.isOpened(): raise ValueError(f'Cannot open {source}')
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    header_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    roi,box_count=crop_box(boxes,width,height)
    n=min(len(bvp),header_count)-SKIP
    if n!=int(row['available_frames_after_skip']): raise ValueError(f'Eligibility length mismatch {key}: {n}')
    crops={s:np.lib.format.open_memmap(dest/f'crop_{s}.npy',mode='w+',dtype=np.uint8,shape=(n,s,s,3)) for s in SIZES}
    index=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if SKIP<=index<SKIP+n:
                x1,y1,x2,y2=roi
                face=cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)
                for size,arr in crops.items():
                    arr[index-SKIP]=cv2.resize(face,(size,size),interpolation=cv2.INTER_AREA)
            index+=1
    finally: cap.release()
    if index!=header_count: raise ValueError(f'Decoded/header frame mismatch {key}: {index}/{header_count}')
    for arr in crops.values(): arr.flush()
    waveform=np.asarray(bvp[SKIP:SKIP+n],dtype=np.float64)
    np.save(dest/'bvp.npy',waveform)
    stats={'schema_version':'compact-v1','session':key,'participant_key':row['participant_key'],'task':row['task'],
           'frames':n,'decoded_frames':index,'video_frames_header':header_count,'bvp_rows':len(bvp),
           'skip_frames':SKIP,'common_eval_frames':int(row['common_eval_frames']),'roi':roi,
           'bbox_rows':len(boxes),'valid_bbox_rows':box_count,'bbox_policy':'median of all finite valid released boxes; 1.5 enlargement',
           'sizes':{str(s):session_stats(arr) for s,arr in crops.items()},
           'label':{'mean':float(waveform.mean()),'std':float(waveform.std()),'diff_std':float(np.diff(waveform).std())},
           'inputs':{'relative_session':str(source.relative_to(Path(root))),'bvp_sha256':digest(source/'BVP_synchronized.csv'),
                     'bbox_sha256':digest(source/'face_bboxes.csv')},
           'moment_precision':'float32 frame/difference arithmetic; float64 accumulated population moments',
           'cache_provenance':provenance,'seconds':round(time.time()-started,3)}
    tmp=complete.with_suffix('.json.tmp');tmp.write_text(json.dumps(stats,indent=2)+'\n');tmp.replace(complete)
    return {'session':key,'status':'created','frames':n,'seconds':stats['seconds']}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--eligibility',type=Path,required=True)
    parser.add_argument('--release-manifest',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--sessions',nargs='*')
    args=parser.parse_args()
    rows=list(csv.DictReader(args.eligibility.open()))
    def yes(x): return str(x).lower() in {'true','1'}
    rows=[r for r in rows if yes(r['eligible']) and (not args.sessions or r['session'] in args.sessions)]
    args.out.mkdir(parents=True,exist_ok=True)
    provenance={'release_manifest_sha256':digest(args.release_manifest),
                'eligibility_sha256':digest(args.eligibility),'builder_sha256':digest(Path(__file__))}
    started=time.time();results=[];errors=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(build_one,(str(args.root),str(args.out),row,provenance)):row['session'] for row in rows}
        for i,future in enumerate(as_completed(pending),1):
            try: result=future.result();results.append(result)
            except Exception as exc:
                result={'session':pending[future],'status':'error','error':str(exc)};errors.append(result)
            print(json.dumps({'done':i,'total':len(rows),**result}),flush=True)
    report={'requested_sessions':len(rows),'completed_sessions':len(results),'errors':errors,
            'wall_seconds':round(time.time()-started,3),'workers':args.workers,'eligibility_sha256':digest(args.eligibility)}
    suffix='pilot' if args.sessions else 'all'
    (args.out/f'build_report_{suffix}.json').write_text(json.dumps(report,indent=2)+'\n')
    if errors: raise SystemExit(1)
    if not args.sessions:
        with (args.out/'manifest.csv').open('w',newline='') as f:
            fields=['session','participant_key','task','frames','start_frame','common_eval_frames']
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for r in rows: w.writerow({k:r[k] for k in ['session','participant_key','task','common_eval_frames']}|{'frames':r['available_frames_after_skip'],'start_frame':SKIP})
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
