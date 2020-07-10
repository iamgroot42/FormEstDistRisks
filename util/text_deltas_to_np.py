import numpy as np
import utils
import sys

src_path = sys.argv[1]
dst_path = sys.argv[2]
deltas  = utils.get_sensitivities(src_path)
np_deltas = np.array(deltas)
np.save(dst_path, np_deltas)
