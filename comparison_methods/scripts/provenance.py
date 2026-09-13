from pathlib import Path
import hashlib
import json
import subprocess

base=Path(__file__).resolve().parents[1]
records=[]
for name,url in [('SeReNet','https://github.com/kimchange/SeReNet'),('VCD-Net','https://github.com/feilab-hust/VCD-Net')]:
    folder=base/'upstream'/name
    sha=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
    licenses=[str(p.relative_to(folder)) for p in folder.rglob('*LICENSE*') if '.git' not in p.parts]
    record={'name':name,'url':url,'commit':sha,'license_files':licenses,'license_status':'provided' if licenses else 'no standalone license file in checked-out upstream source'}
    if name=='SeReNet':
        record['model_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/'save/month_202505/config_tag/code/models').glob('*.py')}
    records.append(record)
(base/'manifests/sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
