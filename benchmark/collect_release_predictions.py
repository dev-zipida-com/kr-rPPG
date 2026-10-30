#!/usr/bin/env python3
"""Score own prediction pickles on fixed common intervals and raw-BVP references.

Model-specific cached label HR is never used as the common reference. Run on
trusted, locally generated pickles only. Per-session output belongs in WORK.
"""
from pathlib import Path
import argparse, csv, hashlib, io, json, pickle, re, sys
import numpy as np
from scipy.signal import butter, filtfilt, periodogram
from common import TOOLBOX, MODELS, import_toolbox

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class CPUUnpickler(pickle.Unpickler):
    """Read trusted prediction tensors on CPU regardless of their saved device."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            import torch
            return lambda payload: torch.load(io.BytesIO(payload), map_location='cpu')
        return super().find_class(module, name)

def predict_hr(signal,fs,diff):
    from evaluation.post_process import _detrend
    signal=np.asarray(signal,dtype=np.float64)
    if not np.isfinite(signal).all(): raise ValueError('Non-finite model prediction')
    if diff: signal=np.cumsum(signal)
    signal=_detrend(signal,100)
    b,a=butter(1,[0.6*2/fs,3.3*2/fs],btype='bandpass')
    signal=filtfilt(b,a,signal)
    f,p=periodogram(signal,fs=fs,nfft=1<<(len(signal)-1).bit_length(),detrend=False)
    mask=(f>=0.6)&(f<=3.3)
    return float(60*f[mask][np.argmax(p[mask])])

def flatten(chunks):
    indexes=sorted(int(k) for k in chunks)
    if indexes!=list(range(len(indexes))): raise ValueError('Missing or duplicate chronological prediction chunks')
    pieces=[]
    for i in indexes:
        a=chunks[i] if i in chunks else chunks[str(i)]
        if hasattr(a,'detach'): a=a.detach().cpu().numpy()
        pieces.append(np.asarray(a).reshape(-1))
    return np.concatenate(pieces)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True,help='The single family directory, e.g. WORK/runs/kr_kfold')
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--oof',action='store_true')
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS),
                        help='Planned model set; use physnet physformer for the paired raw control')
    parser.add_argument('--allow-partial',action='store_true',help='Intermediate monitoring only; final collection must omit this')
    args=parser.parse_args();import_toolbox()
    refs={r['session']:r for r in csv.DictReader(args.reference.open())}
    rx=re.compile(r'^(?P<model>'+'|'.join(MODELS)+r')(?:_fold(?P<fold>[0-4]))?$')
    records=[];source_files=[];seen=set()
    for p in sorted(args.runs.rglob('saved_test_outputs/*_outputs.pickle')):
        info=rx.fullmatch(p.relative_to(args.runs).parts[0])
        if not info: raise ValueError(f'Unknown model run directory: {p}')
        model=info['model'];fold=info['fold'] or ''
        if model not in args.models: raise ValueError(f'Unrequested model in result family: {model}')
        with p.open('rb') as f: data=CPUUnpickler(f).load()
        fs=float(data.get('fs',30))
        label_type=data.get('label_type')
        if label_type not in ['DiffNormalized','DiffOnly','Raw','Standardized']:
            raise ValueError(f'Unexpected label/prediction type {label_type}')
        for key,chunks in sorted(data['predictions'].items()):
            key=str(key)
            if key not in refs: raise ValueError(f'Unexpected prediction session {key}')
            ref=refs[key];length=int(ref['common_eval_frames'])
            if args.oof and fold!=str(ref['fold']): raise ValueError(f'Wrong held-out fold {key} {fold}')
            if abs(float(ref.get('fs',ref.get('sampling_rate_hz',30)))-fs)>1e-8: raise ValueError(f'Sampling rate mismatch {key}')
            unique=(model,fold,key)
            if unique in seen: raise ValueError(f'Duplicate prediction {unique}')
            seen.add(unique)
            prediction=flatten(chunks)
            if len(prediction)<length: raise ValueError(f'Too-short prediction {unique}: {len(prediction)} < {length}')
            estimate=predict_hr(prediction[:length],fs,label_type in ['DiffNormalized','DiffOnly'])
            target=float(ref['reference_hr'])
            records.append({'family':args.runs.name,'model':model,'fold':fold,'session':key,
                'participant_key':ref.get('participant_key',''),'task':ref.get('task',''),
                'pred_hr':estimate,'reference_hr':target,'abs_error':abs(estimate-target),
                'common_eval_frames':length,'predicted_frames':len(prediction),'fs':fs})
        source_files.append({'path':str(p),'sha256':sha(p),'sessions':len(data['predictions'])})
        print(f'{p.parent.parent.name}: {len(data["predictions"])} sessions',flush=True)
    if not records: raise ValueError('No prediction files found')
    if not args.allow_partial:
        for model in args.models:
            selected=[r for r in records if r['model']==model]
            if args.oof:
                if len(selected)!=len(refs) or {r['session'] for r in selected}!=set(refs):
                    raise ValueError(f'Incomplete OOF coverage for {model}')
            else:
                folds={r['fold'] for r in selected}
                if folds not in [{''},set('01234')]: raise ValueError(f'Incomplete folds for {model}: {folds}')
                for fold in folds:
                    if {r['session'] for r in selected if r['fold']==fold}!=set(refs):
                        raise ValueError(f'Incomplete test coverage for {model} fold {fold}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    provenance={'reference_path':str(args.reference),'reference_sha256':sha(args.reference),
                'scorer_sha256':sha(__file__),'sources':source_files,'rows':len(records),'oof':args.oof,
                'models':args.models,
                'partial_monitoring':args.allow_partial,'model_specific_label_hr_used':False,
                'postprocessing_precision':'float64; raw BVP reference on fixed common interval'}
    args.output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'rows':len(records),'output':str(args.output)}),flush=True)

if __name__=='__main__': main()
