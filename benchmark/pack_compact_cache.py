#!/usr/bin/env python3
"""Pack completed compact crops into independently decoded zlib256-frame blocks.

Retains every source NPY. Full packing requires the builder's complete manifest.
No separate full readback pass: zlib checksums, indexed lengths, source fingerprints,
and a streaming raw-array SHA support integrity. Use --sessions only for small pilots.
"""
import argparse,csv,hashlib,json,os,shutil,time,zlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from common import import_toolbox
import_toolbox()
from dataset.data_loader.PackedCache import BLOCK_FRAMES,SCHEMA,source_fingerprint,PackedFrames

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pack_one(job):
    folder,size=job;folder=Path(folder);source=folder/f'crop_{size}.npy'
    packed=source.with_suffix('.zlib');index_path=source.with_suffix('.zlib.json')
    if index_path.exists():
        reader=PackedFrames(source);reader.close()
        return {'session':folder.name,'size':size,'status':'reused','index_sha256':sha(index_path),
                'raw_bytes':json.loads(index_path.read_text())['raw_bytes'],'packed_bytes':packed.stat().st_size}
    if shutil.disk_usage(folder).free < 20 * 1024**3:raise RuntimeError('Packing preserves at least20GiB free space')
    before=source_fingerprint(source)
    if packed.exists():raise RuntimeError(f'Pack exists without complete index: {packed}')
    arr=np.load(source,mmap_mode='r',allow_pickle=False)
    if arr.dtype!=np.uint8 or arr.ndim!=4 or arr.shape[1:]!=(size,size,3):raise ValueError('Unexpected compact source')
    temp=packed.with_suffix('.zlib.part');temp_index=index_path.with_suffix('.json.part')
    blocks=[];digest=hashlib.sha256();started=time.time();offset=0
    with temp.open('xb') as out:
        for start in range(0,len(arr),BLOCK_FRAMES):
            view=arr[start:start+BLOCK_FRAMES];raw=view.tobytes();digest.update(raw)
            encoded=zlib.compress(raw,level=1);out.write(encoded)
            blocks.append({'start_frame':start,'frames':len(view),'offset':offset,'compressed_bytes':len(encoded),'raw_bytes':len(raw)})
            offset+=len(encoded)
        out.flush();os.fsync(out.fileno())
    if source_fingerprint(source)!=before:raise RuntimeError('Source changed during packing')
    index={'schema_version':SCHEMA,'codec':'zlib','level':1,'block_frames':BLOCK_FRAMES,'dtype':'uint8',
           'shape':list(arr.shape),'raw_bytes':int(arr.size),'packed_bytes':offset,'blocks':blocks,
           'source_npy_name':source.name,'source_fingerprint':before,'raw_array_sha256':digest.hexdigest(),
           'stats_sha256':sha(folder/'stats.json'),
           'packer_sha256':sha(__file__),'reader_sha256':sha(Path(__file__).resolve().parents[1]/'third_party/rPPG-Toolbox/dataset/data_loader/PackedCache.py'),
           'verification':'zlib checksum per block on demand; index size/offset/count validation; source selected-byte fingerprint and one-pass raw-array SHA; no full readback'}
    temp_index.write_text(json.dumps(index,indent=2)+'\n');temp.replace(packed);temp_index.replace(index_path)
    return {'session':folder.name,'size':size,'status':'created','index_sha256':sha(index_path),
            'raw_bytes':int(arr.size),'packed_bytes':offset,'seconds':round(time.time()-started,3)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--sizes',type=int,nargs='+',default=[36,72,128]);p.add_argument('--sessions',nargs='+')
    a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('Packing workers must be1–4')
    if not set(a.sizes)<={36,72,128}:raise ValueError('Only existing compact crop sizes supported')
    manifest=a.root/'manifest.csv'
    if a.sessions:session_names=sorted(set(a.sessions))
    else:
        if not manifest.is_file():raise ValueError('Full packing waits for complete builder manifest')
        rows=list(csv.DictReader(manifest.open()));session_names=sorted(r['session'] for r in rows)
        if len(session_names)!=len(set(session_names)):raise ValueError('Duplicate manifest sessions')
    for session in session_names:
        if not (a.root/session/'stats.json').is_file():raise ValueError(f'Incomplete source session: {session}')
    jobs=[(str(a.root/s),size) for s in session_names for size in a.sizes]
    results=[];errors=[];started=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(pack_one,j):j for j in jobs}
        for f in as_completed(futures):
            try:
                r=f.result();results.append(r);print(json.dumps({'done':len(results),'total':len(jobs),**r}),flush=True)
            except Exception as e:
                errors.append({'job':futures[f],'error':repr(e)});print(json.dumps(errors[-1]),flush=True)
    report={'schema_version':SCHEMA,'root':str(a.root.resolve()),'sessions':session_names,'sizes':a.sizes,'block_frames':BLOCK_FRAMES,
            'raw_bytes':sum(r['raw_bytes'] for r in results),'packed_bytes':sum(r['packed_bytes'] for r in results),
            'source_manifest_sha256':sha(manifest) if manifest.exists() else None,'packer_sha256':sha(__file__),
            'results':sorted(results,key=lambda r:(r['session'],r['size'])),'errors':errors,'seconds':time.time()-started,
            'source_npy_retained':True,'full_manifest':not bool(a.sessions)}
    if not a.sessions and not errors:
        target=a.root/'packed_cache_manifest.json';temp=target.with_suffix('.json.part');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(target)
    else:
        name='packed_cache_pilot.json' if a.sessions else 'packed_cache_failed.json'
        (a.root/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'complete':not errors,'raw_bytes':report['raw_bytes'],'packed_bytes':report['packed_bytes'],'seconds':report['seconds']}),flush=True)
    return 1 if errors else 0
if __name__=='__main__':raise SystemExit(main())
