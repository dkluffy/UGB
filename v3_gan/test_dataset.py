# %%

from utils import loadpickle

# %%
pk = loadpickle("dist_pk")

# %%
pk[0]

# %%
lb=pk[1]

# %%
mask=lb>0

# %%
dist=pk[0]

# %%
mask.shape,lb.shape,dist.shape

# %%
mask=mask.flatten()

# %%
mask.shape


# %%
dist[mask]

# %%
import numpy as np
ind = np.argsort(dist)[:10]

# %%
dist[ind],lb[ind],ind

# %%
