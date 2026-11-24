"""GPT-owned sequential continuation of legacy Q7; one numerical child per hold."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/path/to/eegaudit')
OUT=ROOT/'outputs/AGENT_LOOP_20261120'
def record(event):
    event['timestamp']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat()
    event['ownership']='GPT launched'
    event['allocation']=os.environ.get('SLURM_JOB_ID')
    event['step']=os.environ.get('SLURM_STEP_ID')
    with (OUT/'JOBS.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--subs',required=True);a=ap.parse_args()
    for s in a.subs.split(','):
        result=OUT/f'results/q7/fm_probe_reve_{s}.json'
        if result.exists():
            record(dict(event='skipped',purpose='Q7 existing GPT full result',subject=s,output_path=str(result)));continue
        cmd=[sys.executable,'-I','-u',str(ROOT/'experiments/agent_loop_20261120/q7_reve_probe.py'),'--model','reve','--subject',s]
        log=OUT/f'logs/q7_reve_{s}_full.log'
        start=datetime.datetime.now().astimezone().isoformat();t0=time.monotonic()
        record(dict(event='started',purpose='Q7 full REVE official train/test continuation',subject=s,command=cmd,start=start,end=None,exit_code=None,runtime_s=None,output_path=str(log),result_path=str(result)))
        with log.open('w') as f:
            try:
                run=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
                code=run.returncode
            except subprocess.TimeoutExpired:
                code=124
        record(dict(event='completed',purpose='Q7 full REVE official train/test continuation',subject=s,command=cmd,start=start,end=datetime.datetime.now().astimezone().isoformat(),exit_code=code,runtime_s=time.monotonic()-t0,output_path=str(log),result_path=str(result)))
        if code or not result.exists():
            raise SystemExit(code or 1)
        r=json.loads(result.read_text())
        if r['n_train']!=16540:
            record(dict(event='technical_stop',purpose='Q7 unexpected official train count',subject=s,n_train=r['n_train']));raise SystemExit(2)
        print(s,'COMPLETED',time.monotonic()-t0,flush=True)

if __name__=='__main__':main()
