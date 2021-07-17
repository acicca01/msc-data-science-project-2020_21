#Create dummy music collection

import pandas as pd
import numpy as np

plylst_id = np.arange(0,10,1)
mysongs=[]
for i in range(10):
    mysongs.append(np.random.choice(14,10,replace = False))

dummy = pd.DataFrame()
dummy["plylst_id"] = pd.Series(plylst_id)
dummy["songs"] = pd.Series(mysongs)




