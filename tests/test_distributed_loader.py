"""Exercise actual sharding and resume code without loading a model."""
import hashlib,io,json,random
from unittest.mock import patch
from distributed_ocr import RankStream
rows=[{'id':i,'split':'train','image':{'key':'image','offset':0,'size':1}} for i in range(24)]
raw=b'\n'.join(json.dumps(r).encode() for r in rows)
class FakeS3:
    def get_object(self,**kw): return {'Body':io.BytesIO(raw if kw['Key']=='manifest' else b'x')}
def loader(state):
    x=RankStream.__new__(RankStream); x.s3=FakeS3(); x.reports=[{'manifest':{'key':'manifest','sha256':hashlib.sha256(raw).hexdigest()}}]
    x.encode=lambda row,raw:row['id']; x.load_state_dict(state); return x
expected=rows.copy(); random.Random(42).shuffle(expected); expected=[r['id'] for r in expected]
state={'shard':0,'cursor':8,'count':8}
for start,world in [(8,4),(16,2)]:
    seen=[]; states=[]
    for rank in range(world):
        with patch('distributed_ocr.dist.get_rank',return_value=rank),patch('distributed_ocr.dist.get_world_size',return_value=world):
            x=loader(state); it=iter(x); seen.extend([next(it) for _ in range(8//world)]); states.append(x.state_dict())
    assert len(set(seen))==8 and set(seen)==set(expected[start:start+8])
    state=max(states,key=lambda s:s['count']); assert state['count']==start+8
print('DISTRIBUTED_LOADER_RESUME_PASSED')
