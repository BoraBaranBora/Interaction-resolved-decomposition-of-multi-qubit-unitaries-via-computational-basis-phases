import json
from pathlib import Path
from control_optimization.config import load_control_config

def test_noise_robust_fields_load(tmp_path: Path):
    path = tmp_path / 'control.json'
    path.write_text(json.dumps({
        'gate':'zzz','duration_ns':100.0,'basis_size':3,'output_dir':'results/test',
        'pulse_parameterization':'direct_fourier',
        'trajectory_sample_stride':4,
        'objective_weights':{'electron_dephasing_exposure':0.03}
    }), encoding='utf-8')
    cfg=load_control_config(path)
    assert cfg.trajectory_sample_stride == 4
    assert cfg.objective_weights.electron_dephasing_exposure == 0.03
