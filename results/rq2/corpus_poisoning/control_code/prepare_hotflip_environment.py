from pathlib import Path
import getpass,paramiko,shlex,sys
sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
password=getpass.getpass('SSH password for 35564 (not saved): ')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('connect.nmb2.seetacloud.com',port=35564,username='root',password=password,look_for_keys=False,allow_agent=False,timeout=20);password='';c.get_transport().set_keepalive(20)
root='/root/autodl-tmp/corpus_hotflip_baseline_20261031'
def run(cmd):
    _,o,e=c.exec_command(cmd+' 2>&1',timeout=1800)
    for line in o:print(line.rstrip(),flush=True)
    if o.channel.recv_exit_status():raise RuntimeError('Environment setup failed')
run('mkdir -p '+shlex.quote(root))
run('/root/miniconda3/bin/python -m venv --system-site-packages '+shlex.quote(root+'/venv'))
run(shlex.quote(root+'/venv/bin/python')+' -m pip install --index-url https://pypi.org/simple "numpy==1.26.4" "transformers==4.44.2" "scikit-learn==1.5.2"')
probe="import torch,numpy,transformers,sklearn; print('ISOLATED_ENV_READY',torch.__version__,numpy.__version__,transformers.__version__,sklearn.__version__); print('CUDA_AVAILABLE',torch.cuda.is_available())"
run(shlex.quote(root+'/venv/bin/python')+' -c '+shlex.quote(probe));c.close()
