import os,sys
from pathlib import Path
currentdir = Path(os.getcwd())
sys.path.insert(0, os.path.dirname(currentdir.parent))
sys.path.insert(0, os.path.dirname(currentdir))
print(str(sys.path[0:3]))

import pandas as pd
from varclushi import VarClusHi

data = pd.read_csv('examples/data/pigs.csv')
vc = VarClusHi(data, maxeigval2=1, maxclus=None)
vc.varclus()
vc.info