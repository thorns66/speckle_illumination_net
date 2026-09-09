"""Recompute the actual brightness-sweep inputs and check the gain-one reference."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from tools.three_way_experiment import OUTPUT,PRIORITY,DATA,sha256,write_json
import tools.priority_validation_analysis as pa

def run():
    rows=[]
    for sample,source in [('P09',DATA/'P09'),('points_z060_r01',PRIORITY/'generated/points_z060_r01'),
                          ('lines_z060_r01',PRIORITY/'generated/lines_z060_r01')]:
        with h5py.File(source/'subsets/subset_01.mat','r') as f:
            indices=np.asarray(f['input_indices']).reshape(-1).astype(int)
            old_mean=pa._mat_yxz(f,'physics_mean_raw')
            old_taylor=pa._mat_yxz(f,'physics_taylor_sqrt_float')
        frames=[]
        for index in indices:
            with h5py.File(source/'sensor_frames'/f'frame_{index:03d}.mat','r') as f:
                frames.append(pa._mat_yx(f,'sensor_pre_detector'))
        original=np.stack(frames)
        for gain in [.1,.5,1.,2.]:
            path=OUTPUT/'brightness_inputs'/sample/f'gain_{gain:g}.mat'
            with h5py.File(path,'r') as f:
                raw=np.asarray(f['input_frames']).transpose(0,2,1)
                mean=pa._mat_yx(f,'input_mean');var=pa._mat_yx(f,'input_variance')
                taylor=pa._mat_yxz(f,'taylor_raw');sqrt=pa._mat_yxz(f,'f_var');gmean=pa._mat_yxz(f,'g_mean')
            expected=(original.astype(np.float64)*gain).astype(np.float32)
            row={'sample_id':sample,'gain':gain,'raw_exact':bool(np.array_equal(raw,expected)),
                 'mean_exact':bool(np.array_equal(mean,raw.mean(0,dtype=np.float64).astype(np.float32))),
                 'variance_nminus1_exact':bool(np.array_equal(var,raw.var(0,ddof=1,dtype=np.float64).astype(np.float32))),
                 'sqrt_once_relative_l2':float(np.linalg.norm(sqrt-np.sqrt(taylor))/max(np.linalg.norm(sqrt),1e-30)),
                 'sha256':sha256(path)}
            if gain==1:
                row['mean_RL3_reproduction_relative_l2']=float(np.linalg.norm(gmean-old_mean)/max(np.linalg.norm(old_mean),1e-30))
                row['taylor_RL3_reproduction_relative_l2']=float(np.linalg.norm(sqrt-old_taylor)/max(np.linalg.norm(old_taylor),1e-30))
                assert row['mean_RL3_reproduction_relative_l2']<1e-4 and row['taylor_RL3_reproduction_relative_l2']<1e-4
            assert row['raw_exact'] and row['mean_exact'] and row['variance_nminus1_exact'] and row['sqrt_once_relative_l2']<1e-6
            rows.append(row)
    write_json(OUTPUT/'brightness_inputs/verification.json',{'passed':True,'cases':rows,
          'all_means_and_unbiased_variances_recomputed':True,'gain_one_reproduces_existing_RL3':True})
    print('Brightness input verification passed for all 12 cases.',flush=True)

if __name__=='__main__':run()
