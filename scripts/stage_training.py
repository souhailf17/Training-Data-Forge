"""Prepare owned PVC inputs, audit manifests, and inspect installed training APIs."""
import hashlib, inspect, json, os
from pathlib import Path
import boto3
from botocore.config import Config

ROOT = Path('/work/q35ocr-20260921-p560')
BUCKET = 'qwen35-9b-ocr-finetuning'
PROJECT = 'q35ocr-20260921-p560'

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ['model', 'data', 'logs', 'checkpoints', 'cache']:
        (ROOT/name).mkdir(exist_ok=True)
    s3 = None
    for endpoint in [os.environ['AWS_ENDPOINT_URL'], 'http://q35ocr-20260921-p560-s3-relay.mtrna-ns1.svc:9000']:
        try:
            candidate = boto3.client('s3', endpoint_url=endpoint, region_name='us-east-1', config=Config(connect_timeout=5, read_timeout=60, retries={'max_attempts':1}, s3={'addressing_style':'path'}))
            owner = json.loads(candidate.get_object(Bucket=BUCKET, Key='documentation/project-owner.json')['Body'].read())
            if PROJECT not in json.dumps(owner):
                raise RuntimeError('Bucket ownership mismatch')
            s3 = candidate
            (ROOT/'endpoint.txt').write_text(endpoint)
            break
        except Exception as exc:
            print('Endpoint check:', endpoint, type(exc).__name__, getattr(exc,'response',{}).get('Error',{}), flush=True)
    if s3 is None:
        raise RuntimeError('Project storage unreachable')
    summary = json.loads(s3.get_object(Bucket=BUCKET, Key='dataset/manifests/ocr-manifest-v1/COMPLETE.json')['Body'].read())
    assert len(summary['shards']) == 412
    checked = 0
    for report in summary['shards']:
        for name in ['manifest', 'rejections']:
            item = report[name]
            head = s3.head_object(Bucket=BUCKET, Key=item['key'])
            assert head['ContentLength'] == item['bytes']
            assert head['Metadata']['sha256'] == item['sha256']
            checked += 1
    (ROOT/'data/COMPLETE.json').write_text(json.dumps(summary))
    print('Prepared object metadata verified:', checked, flush=True)
    tree = json.loads(Path('/project/model-tree.json').read_text())
    for item in tree:
        if item['type'] != 'file': continue
        path = item['path']
        record = json.loads(s3.get_object(Bucket=BUCKET, Key='dataset/manifests/transfers/model/'+hashlib.sha256(path.encode()).hexdigest()+'.json')['Body'].read())
        assert record['status'] in ['verified','verified-existing']
        target = ROOT/'model'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        def digest(p):
            with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
        if not target.exists() or digest(target) != record['checksum']:
            temp = target.with_suffix(target.suffix+'.partial')
            s3.download_file(BUCKET, 'model/base/'+path, str(temp))
            assert temp.stat().st_size == item['size'] and digest(temp) == record['checksum']
            temp.replace(target)
        print('Model verified on PVC:', path, flush=True)
    rows = []; groups = set(); counts = {'train':0, 'validation':0, 'test':0}
    for report in summary['shards']:
        body = s3.get_object(Bucket=BUCKET, Key=report['manifest']['key'])['Body']
        try:
            for line in body.iter_lines():
                r = json.loads(line)
                if counts[r['split']] >= 8 or r['group_id'] in groups: continue
                im = r['image']; start = im['offset']
                data = s3.get_object(Bucket=BUCKET, Key=im['key'], Range=f"bytes={start}-{start+im['size']-1}")['Body'].read()
                assert hashlib.sha256(data).hexdigest() == im['sha256']
                assert hashlib.sha256(r['text'].encode()).hexdigest() == r['text_sha256']
                dest = ROOT/'data'/f'{len(rows):03d}.image'
                dest.write_bytes(data); r['local_image'] = str(dest)
                rows.append(r); groups.add(r['group_id']); counts[r['split']] += 1
                if min(counts.values()) == 8: break
        finally: body.close()
        if min(counts.values()) == 8: break
    assert min(counts.values()) == 8
    (ROOT/'data/smoke.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    import importlib.metadata as metadata
    from arsenal.config.optimizer import OptimizerConfig
    from arsenal.config.distributed import DistributedConfig
    from arsenal.config.dataloader import BaseDataloaderConfig
    from lr_scheduler_basic import CosineSchedulerConfig
    from arsenal.model import ArsenalModule
    from arsenal.trainer import BaseTrainer
    with (ROOT/'logs/training-api.txt').open('w') as out:
        for cls in [OptimizerConfig, DistributedConfig, BaseDataloaderConfig, CosineSchedulerConfig, ArsenalModule, BaseTrainer]:
            out.write('\n'+inspect.getsource(cls))
        out.write('\nPACKAGES\n'+json.dumps({d.metadata['Name']:d.version for d in metadata.distributions()}))
    result = {'status':'passed', 'model_files':len([i for i in tree if i['type']=='file']), 'prepared_objects_checked':checked, 'smoke_samples':counts}
    (ROOT/'logs/staging-result.json').write_text(json.dumps(result))
    print(json.dumps(result), flush=True)

if __name__ == '__main__': main()
