from functools import partial
import pandas as pd
import json
import json
import pickle
import os.path as osp
import csv
import datetime

ROOT = 'InternVL2-8B-llava-s2/'


import json
import datetime

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)
        
# Function to get the current time as a string
def timestr(second=True, minute=False):
    s = datetime.datetime.now().strftime('%Y%m%d%H%M%S')[2:]
    if second:
        return s
    elif minute:
        return s[:-2]
    else:
        return s[:-4]

def istype(s, type):
    if isinstance(s, type):
        return True
    try:
        return isinstance(eval(s), type)
    except Exception as _:
        return False

def load(f):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](f)

def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)



# get metadata
def down_mmbench():
    download_file('http://opencompass.openxlab.space/assets/mmbench/mmbench-data.json')

def get_metas():
    data = load('mmbench-data-local.json')
    metas = list(data['META_MAP'].values())
    for item in metas:
        item.pop('key')
    return metas

def upper_case(m):
    if m not in ['Overall', 'CP', 'FP-C', 'FP-S', 'AR', 'LR', 'RR']:
        ms = m.split('_')
        ms = [x[0].upper() + x[1:] for x in ms]
        return ' '.join(ms)
    return m

# down_mmbench()
metas = get_metas()
meta_map = {x['Method'][0]: x for x in metas}
print(meta_map.keys())



MMBenchData = load('mmbench-data.json')

def read_float(f):
    num = float(f) * 100
    num = f'{num:.1f}'
    return float(num)

POP_KEYS = ['Method', 'Parameters', 'Language Model', 'Vision Model', 'Org', 'Time', 'Verified', 'OpenSource', 'key']
MMBench_KEYS = ['Overall', 'CP', 'FP-C', 'FP-S', 'LR', 'AR', 'RR']

def MMBench(model, split='DEV_EN', version='V10'):
    try:
        if 'TEST' in split:
            # Try to pre-fetch first
            key = 'MMBench_' + split + ('_V11' if version == 'V11' else '') + '_Data'
            data = MMBenchData[key]
            model_name = model_map[model]
            res = [item for item in data if item['Method'][0] == model_name][0]
            res = {k: v for k, v in res.items() if k not in POP_KEYS}
            if version == 'V10':
                res = {k: v for k, v in res.items() if k in MMBench_KEYS}
            res = {upper_case(k): v for k, v in res.items()}
            return res
    except:
        pass
    
    assert version in ['V10', 'V11']
    assert split in ['DEV_EN', 'DEV_CN', 'TEST_EN', 'TEST_CN']
    is_dev = 'DEV' in split
    try1 = f'{ROOT}/{model}/{model}_MMBench_{split}_V11_acc.csv'
    try2 = f'{ROOT}/{model}/{model}_MMBench_V11_acc.csv' if 'EN' in split else f'{ROOT}/{model}/{model}_MMBench_CN_V11_acc.csv'
    if version == 'V10':
        try1 = try1.replace('_V11_', '_')
        try2 = try2.replace('_V11_', '_')
    assert osp.exists(try1) or osp.exists(try2)
    data = load(try1) if osp.exists(try1) else load(try2)
    data = data[data['split'] == ('dev' if is_dev else 'test')].iloc[0]
    keys = [k for k in data.keys() if k != 'split']
    if version == 'V10':
        keys = MMBench_KEYS
    
    return {upper_case(k): read_float(data[k]) for k in keys}

def CCBench(model):
    try:
        data = MMBenchData['CCBench_Data']
        model_name = model_map[model]
        res = [item for item in data if item['Method'][0] == model_name][0]
        res = {k: v for k, v in res.items() if k not in POP_KEYS}
        res = {upper_case(k): v for k, v in res.items()}
        return res
    except:
        pass
    
    pth = f'{ROOT}/{model}/{model}_CCBench_acc.csv'
    data = load(pth)
    keys = [k for k in data.keys() if k != 'split']
    return {upper_case(k): read_float(data[k]) for k in keys}

def SEEDBench(model):
    acc = f'{ROOT}/{model}/{model}_SEEDBench_IMG_acc.csv'
    data = load(acc)
    data = {k: data[k][0] * 100 for k in data if k != 'split'}
    official = {
        'GPT-4v (detail: low)': '69.1 (Gen)', 'LLaVA-v1.5-13B': '68.2 (Gen)',
        'ShareGPT4V-13B': '70.8 (Gen)', 'ShareGPT4V-7B': '69.7 (Gen)',
        'InternLM-XComposer-VL': '66.9 (PPL)', 'mPLUG-Owl2': '64.1 (Not Given)',
        'Qwen-VL-Chat': '65.4 (PPL)', 'Qwen-VL': '62.3 (PPL)',
        'IDEFICS-80B-Instruct': '53.2 (Not Given)', 'IDEFICS-9B-Instruct': '44.5 (Not Given)',
        'InstructBLIP-7B': '58.8 (PPL)', 'MiniGPT-4-v1-7B': '47.4 (PPL)', 'OpenFlamingo v2': '42.7 (PPL)'
    }
    data['Overall (official)'] = official[model_map[model]] if model_map[model] in official else 'N/A'
    return data
    
def MME(model):
    data = load(f'{ROOT}/{model}/{model}_MME_score.csv')
    result = {upper_case(k): data[k][0] for k in data}
    ret = {'Perception': result.pop('Perception'), 'Cognition': result.pop('Reasoning')}
    base = {'Overall': ret['Perception'] + ret['Cognition']}
    base.update(ret)
    base.update(result)
    return base




def MMVet(model):
    data = load(f'{ROOT}/{model}/{model}_MMVet_gpt-4-turbo_score.csv')
    ret = {upper_case(x): y for x, y in zip(data['Category'], data['acc'])}
    official = {
        'GeminiProVision': '64.3±0.4', 'GPT-4v (detail: low)': '60.2±0.3',
        'LLaVA-v1.5-13B': '36.3±0.2', 'LLaVA-v1.5-7B': '31.1±0.2', 'LLaVA-v1-7B': '23.8±0.6',
        'mPLUG-Owl2': '36.3±0.1',
        'ShareGPT4V-13B': '43.1', 'ShareGPT4V-7B': '37.6',
        'InstructBLIP-7B': '26.2±0.2', 'InstructBLIP-13B': '25.6±0.3',
        'OpenFlamingo v2': '24.8±0.2',
        'MiniGPT-4-v1-13B': '24.4±0.4', 'MiniGPT-4-v1-7B': '22.1±0.1',
        'EMU2-Chat': '48.5', 
    }
    ret['Overall (official)'] = official[model_map[model]] if model_map[model] in official else 'N/A'
    return ret

def MMMU(model):
    data = load(f'{ROOT}/{model}/{model}_MMMU_DEV_VAL_acc.csv')
    dims = ['Overall', 'Art & Design', 'Business', 'Science', 'Health & Medicine', 'Humanities & Social Science', 'Tech & Engineering']
    line = data[data['split'] == 'validation']
    return {k: line.iloc[0][k] * 100 for k in dims}

def MathVista(model):
    abbrs = {
    'Overall': 'Overall', 'scientific reasoning': 'SCI', 'textbook question answering': 'TQA', 
    'numeric commonsense': 'NUM', 'arithmetic reasoning': 'ARI', 'visual question answering': 'VQA', 
    'geometry reasoning': 'GEO', 'algebraic reasoning': 'ALG', 'geometry problem solving': 'GPS', 
    'math word problem': 'MWP', 'logical reasoning': 'LOG', 'figure question answering': 'FQA',
    'statistical reasoning': 'STA'}
    data = load(f'{ROOT}/{model}/{model}_MathVista_MINI_gpt-4-turbo_score.csv')
    res = {abbrs[k]: v for k, v in zip(data['Task&Skill'], data['acc'])}
    return res

def HallusionBench(model):
    data = load(f'{ROOT}/{model}/{model}_HallusionBench_score.csv')
    data = data[data['split'] == 'Overall']
    ret = {k: data[k][0] for k in ['aAcc', 'fAcc', 'qAcc']}
    ret['Overall'] = np.mean(list(ret.values()))
    return ret

def LLaVABench(model):
    data = load(f'{ROOT}/{model}/{model}_LLaVABench_score.csv')
    ret = {upper_case(k): v for k, v in zip(data['split'], data['Relative Score (main)'])}
    official = {
        'ShareGPT4V-13B': 79.9, 'ShareGPT4V-7B': 72.6, 
        'LLaVA-v1.5-13B': 70.7, 'LLaVA-v1.5-7B': 63.4,
        'InstructBLIP-7B': 60.9, 'InstructBLIP-13B': 58.2,
    }
    ret['Overall (official)'] = official[model_map[model]] if model_map[model] in official else 'N/A'
    return ret        



def AI2D(model):
    data = load(f'{ROOT}/{model}/{model}_AI2D_TEST_acc.csv')
    ret = {k: data[k][0] * 100 for k in data if k != 'split'}
    return ret

def ScienceQA(model, split='VAL'):
    data = load(f'{ROOT}/{model}/{model}_ScienceQA_{split}_acc.csv')
    ret = {k: data[k][0] * 100 for k in data if k != 'split'}
    return ret

def COCO_VAL(model):
    data = load(f'{ROOT}/{model}/{model}_COCO_VAL_score.json')
    res = {'BLEU-1': data['Bleu'][0], 'BLEU-4': data['Bleu'][3], 'ROUGE-L': data['ROUGE_L'], 'CIDEr': data['CIDEr']}
    return res

def OCRBench(model):
    data = load(f'{ROOT}/{model}/{model}_OCRBench_score.json')
    res = {k: v for k, v in data.items() if k != 'Final Score Norm'}
    return res

def MMStar(model):
    data = load(f'{ROOT}/{model}/{model}_MMStar_acc.csv')
    ret = {k: data[k][0] * 100 for k in data if k != 'split'}
    return ret

def RealWorldQA(model):
    data = load(f'{ROOT}/{model}/{model}_RealWorldQA_acc.csv')
    ret = {k: data[k][0] * 100 for k in data if k != 'split'}
    return ret




def TextVQA_VAL(model):
    data = load(f'{ROOT}/{model}/{model}_TextVQA_VAL_acc.csv')
    ret = {k: data[k][0] for k in data if k != 'split'}
    return ret

def ChartQA_TEST(model):
    data = load(f'{ROOT}/{model}/{model}_ChartQA_TEST_acc.csv')
    ret = {k: data[k][0] for k in data if k != 'split'}
    return ret

def OCRVQA_TESTCORE(model):
    data = load(f'{ROOT}/{model}/{model}_OCRVQA_TESTCORE_acc.csv')
    ret = {k: data[k][0] for k in data if k != 'split'}
    return ret

def POPE(model):
    data = load(f'{ROOT}/{model}/{model}_POPE_score.csv')
    data = data[data['split'] == 'Overall']
    ret = {k: data[k][0] for k in ['Overall', 'acc', 'precision', 'recall']}
    return ret


func_map = {
    'SEEDBench_IMG': SEEDBench, 'CCBench': CCBench, 
    'MMBench_TEST_EN': partial(MMBench, split='TEST_EN', version='V10'), 'MMBench_TEST_CN': partial(MMBench, split='TEST_CN', version='V10'), 
    'MMBench_TEST_EN_V11': partial(MMBench, split='TEST_EN', version='V11'),
    'MMBench_TEST_CN_V11': partial(MMBench, split='TEST_CN', version='V11'), 
    'MME': MME, 'MMVet': MMVet, 'MMMU_VAL': MMMU, 
    'MathVista': MathVista, 'HallusionBench': HallusionBench, 'LLaVABench': LLaVABench, 'AI2D': AI2D, 'COCO_VAL': COCO_VAL, 
    'ScienceQA_VAL': partial(ScienceQA, split='VAL'), 'ScienceQA_TEST': partial(ScienceQA, split='TEST'), 'OCRBench': OCRBench, 'MMStar': MMStar, 
    'RealWorldQA': RealWorldQA,  'TextVQA_VAL': TextVQA_VAL, 'ChartQA_TEST': ChartQA_TEST, 'OCRVQA_TESTCORE': OCRVQA_TESTCORE,  'POPE': POPE
}

def robustify(func, key, log_lvl=2):
    def robust_func(model):
        try:
            res = func(model)
            res = {k: (float(v) if istype(v, float) else v) for k, v in res.items()}
            return res
        except:
            if log_lvl < 3 and key in ['TextVQA_VAL', 'ChartQA_TEST', 'OCRVQA_TESTCORE', 'COCO_VAL']:
                pass
            else:
                print(f'Failed to retrieve results: {func}, {model}')
            return None
    return robust_func

func_map = {k: robustify(v, k) for k, v in func_map.items()}

API_models=[
    'Gemini-1.0-Pro', 'Gemini-1.5-Pro',
    'GPT-4v (1106, detail-low)', 'GPT-4v (1106, detail-high)', 
    'GPT-4v (0409, detail-low)', 'GPT-4v (0409, detail-high)', 
    'GPT-4o (0513, detail-low)', 'GPT-4o (0513, detail-high)',
    'Qwen-VL-Plus', 'Qwen-VL-Max', 'Step-1V', 
    'Claude3-Haiku', 'Claude3-Sonnet', 'Claude3-Opus', 'Claude3.5-Sonnet',
    'RekaFlash', 'RekaEdge', 'RekaCore', 'GLM-4v', 'CongRong',
    'Step-1V-0701', 'InternVL-2.0-Pro'
]

model_map = {
             'InternVL-Chat-V1-2-Plus':'InternVL-Chat-V1.2-Plus',
             'InternVL-Chat-V1-5':'InternVL-Chat-V1.5',
             # 'InternVL-Chat-V1-5-Int':'InternVL-Chat-V1.5-INT',
             'InternVL2-1B' : 'InternVL2-1B',
             'InternVL2-2B' : 'InternVL2-2B',
             'InternVL-Chat-V1-5-Int':'InternVL-Chat-V1.5-2B-INT',
             'InternVL2-8B' : 'InternVL2-8B',
             'InternVL2-8B-Int' : 'InternVL2-8B-Int',
             'InternVL2-8B-Int2' : 'InternVL2-8B-Int2',
             'InternVL2-13B-Int' : 'InternVL2-13B-Int',
             'InternVL2-26B':'InternVL2-26B',
             'InternVL2-26B-Int':'InternVL2-26B-INT',
             'InternVL2-76B':'InternVL2-76B',
             'idefics2_8b':'IDEFICS2-8B',
             'idefics_9b_instruct' : 'IDEFICS-9B-Instruct',
             'idefics_80b_instruct' : 'IDEFICS-80B-Instruct' ,
             'llava_next_yi_34b':'LLaVA-Next-Yi-34B',
             'llava_v1.5_7b':'LLaVA-v1.5-7B',
             'llava_v1.5_13b':'LLaVA-v1.5-13B',
             'llava_v1.5_7b_int':'LLaVA-v1.5-7B-INT',
             'llava_v1.5_13b_int':'LLaVA-v1.5-13B-INT',
             'sharegpt4v_13b':'ShareGPT4V-13B',
             'Phi-3-Vision' : 'Phi-3-Vision',
             'vila_3b' : 'VILA1.5-3B',
             'vila_8b' : 'VILA1.5-8B',
             'vila_13b' : 'VILA1.5-13B',
             'vila_40b' : 'VILA1.5-40B',
             'vila_3b_int' : 'VILA1.5-3B-INT',
             'vila_8b_int' : 'VILA1.5-8B-INT',
             'Mini-InternVL-Chat-2B-V1-5' : 'Mini-InternVL-Chat-2B-V1.5',


            # 
            'llava_onevision_qwen2_0.5b_si' : 'LlaVa-OneVision-0.5B-SI',
            'llava_onevision_qwen2_0.5b_ov' : 'LlaVa-OneVision-0.5B-OV',
            'llava_onevision_qwen2_7b_si' : 'LlaVa-OneVision-7B-SI',
            'llava_onevision_qwen2_7b_ov' : 'LlaVa-OneVision-7B-OV',

            'Phi-3.5-Vision' : 'Phi-3.5-Vision',
            'InternVL2-4B-LOVD-S2': 'InternVL2-4B-LOVD-S2' ,
            'InternVL2-26B-LOVD-S2': 'InternVL2-26B-LOVD-S2',
            # 'xgen-mm-phi3-dpo-r-v1.5':  ,
            # 'xgen-mm-phi3-interleave-r-v1.5': ,
    
             
            }

main = {}

for m in model_map:
    model_name = model_map[m]
    if model_name in meta_map and osp.exists(f'{ROOT}/{m}'):
        meta = meta_map[model_name]
        if model_name in API_models:
            meta['OpenSource'] = 'No'
        main[model_name] = {'META': meta}
        for d in func_map:
            out_file = f'{ROOT}/{m}/{m}_{d}.xlsx'
            ret = func_map[d](m)
            if ret is None:
                continue
            ret = {k: round(v, 1) if isinstance(v, float) else v for k, v in ret.items()}
            main[model_name][d] = ret
    else:
        print(m, model_name)



for name in main: 
    for d in ['TextVQA_VAL', 'ChartQA_TEST', 'OCRVQA_TESTCORE']:
        if d in main[name] and main[name][d]['Overall'] < 10:
            main[name].pop(d)

dump(dict(time=timestr(), results=main), 'OpenVLM.json')
main.keys()
