"""Output-only axis overlay checks; the frozen input-stage reader is reused."""
import json
from pathlib import Path
from .input_axis_residual import read_axis_e8,sha
from .runtime import MODEL_CONFIG

FORMAT='MAMBA2_OUTPUT_AXIS_RESIDUAL_V1'
LABELS=tuple(f'layer{i}.out_proj' for i in range(56))
SHAPE=[4096,8192]
RECIPE={'residual_bits':4,'index_bits':20,'values_per_index':8,'tune_iters':2,
    'damping':.01,'scale_override':.9,'feedback':True,'seed_rule':'1001 + 2*layer'}
PROTOCOL_SHA='3ac0a78c4b93dcc84f2402b39847fd93c5daf1705cbca360c2cdfa84e7e545e4'
INPUT_MANIFEST_SHA='0fabdec5f6713f4530723f1e6a8c0e8b34563edea015441035613e5232347c85'
INPUT_REPORT_SHA='1dd6be0ddb4481e765da2658926ecea92183521ac9862b914fc3d3ba2a80a7eb'


def verify_overlay(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    names={label+'.e8' for label in LABELS}
    if (manifest.get('format')!=FORMAT or manifest.get('complete') is not True
            or manifest.get('model_config')!=MODEL_CONFIG or manifest.get('recipe')!=RECIPE
            or manifest.get('input_manifest_sha256')!=INPUT_MANIFEST_SHA
            or manifest.get('input_evaluation_sha256')!=INPUT_REPORT_SHA
            or manifest.get('binding',{}).get('protocol_sha256')!=PROTOCOL_SHA
            or set(manifest.get('matrices',{}))!=set(LABELS) or set(manifest.get('files',{}))!=names
            or {p.name for p in directory.iterdir()}!=names|{'manifest.json'}):
        raise ValueError('Incomplete or unexpected output-axis overlay inventory/recipe')
    for layer,label in enumerate(LABELS):
        entry=manifest['matrices'][label];path=directory/(label+'.e8');file=manifest['files'][path.name]
        if (entry['file']!=path.name or entry['shape']!=SHAPE or entry['seed']!=1001+2*layer
                or entry['source_key']!=f'backbone.layers.{layer}.mixer.out_proj.weight'
                or entry['index_bits']!=20 or entry['values_per_index']!=8
                or entry['tune_iters']!=2 or entry['damping']!=.01 or entry['scale_override']!=.9
                or entry['feedback'] is not True or entry['disk_roundtrip_fp16_equal'] is not True
                or entry['packed_indices_equal'] is not True):raise ValueError('Output-axis matrix recipe differs')
        if file!={k:entry[k] for k in ('bytes','sha256','decoded_fp16_sha256')}:
            raise ValueError('Output matrix/file ledgers differ')
        if path.stat().st_size!=file['bytes'] or sha(path)!=file['sha256']:raise ValueError('Output-axis file identity differs')
        payload,layout=read_axis_e8(path,SHAPE)
        if layout!=entry['layout'] or payload['axis_residual_amplitude']!=entry['axis_residual_amplitude']:
            raise ValueError('Output-axis header/length metadata differs')
        if len(entry['decoded_fp16_sha256'])!=64:raise ValueError('Missing decoded output identity')
    if sum(row['bytes'] for row in manifest['files'].values())!=manifest['data_file_bytes']:
        raise ValueError('Output overlay byte ledger differs')
    return manifest
