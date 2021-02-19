import sys
import os
import subprocess
from torchbiggraph.train import main


print("MASTER_ADDR::", os.environ['MASTER_ADDR'])

if __name__ == '__main__':
    child = subprocess.Popen(['pgrep', '-f', 'train_wrapper.py'], stdout=subprocess.PIPE, shell=False)
    response = child.communicate()[0]
    running_procs = [int(pid) for pid in response.split()]
    if len(running_procs) > 1:
        for pid in running_procs:
            print('Already running train_wrapper.py is '+ str(pid))

    sys.exit(main())

