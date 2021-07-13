import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description='Submit rebinAndRunLimitsWorker jobs to HTCondor.')
parser.add_argument('--output', required=True, type=str, help="output directory")
parser.add_argument('--channel', required=True, type=str, help="channel")
parser.add_argument('--n-workers', required=True, type=int, help="number of worker jobs to submit")
parser.add_argument('--verbose', required=False, type=int, default=1, help="verbosity")
args = parser.parse_args()

workers_output = os.path.join(args.output, args.channel, "workers")

if not os.path.isdir(workers_output):
    raise RuntimeError("Workers output directory not found. Please, make sure that the server is running.")

run_script = '''#!/bin/bash
cd {}
source env.sh hh
python tools/rebinAndRunLimitsWorker.py --output {} --verbose {}
'''.format(os.getcwd(), workers_output, args.verbose)
script_file = os.path.join(workers_output, 'run_worker.sh')
with open(script_file, 'w') as f:
    f.write(run_script)
os.chmod(script_file, 0o744)

condor_job = '''
executable              = {0}
log                     = {1}/worker.$(ClusterId).$(ProcId).log
output                  = {1}/worker.$(ClusterId).$(ProcId).out
error                   = {1}/worker.$(ClusterId).$(ProcId).err
+JobFlavour             = "tomorrow"
queue {2}
'''.format(script_file, workers_output, args.n_workers)

sub_file = os.path.join(workers_output, 'worker.sub')
with open(sub_file, 'w') as f:
    f.write(condor_job)

subprocess.call(['condor_submit -batch-name {} {}'.format(args.channel, sub_file)], shell=True)
