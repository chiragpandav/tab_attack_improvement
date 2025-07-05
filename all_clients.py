import subprocess
import sys
import os

# state_codes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

state_codes =[0]
for state_code in state_codes:
    subprocess.run(["python3", "run_fed_avg_attacks.py", "--experiment", "0", "--name_state",  str(state_code)])

