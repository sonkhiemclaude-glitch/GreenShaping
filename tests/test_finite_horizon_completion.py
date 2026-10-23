"""Cross-check the new last-tick analytical limit against the release primitive."""
import math
import numpy as np
from greenshaping.shaping.dp_release import release_link

def test_last_tick_overflow_matches_exact_laplace_tail():
 rng=np.random.default_rng(20261026);V=8;scale=16;n=20000
 for ratio in (0,2,8):
  bias=ratio*scale;failures=0
  for _ in range(n):
   _,_,remaining=release_link([(.95,V,0)],[1.0],scale,'arrivals',bias,rng);failures+=remaining>0
  expected=.5*math.exp(-(bias+.5)/scale)
  assert abs(failures/n-expected)<=5*math.sqrt(expected*(1-expected)/n)+1/n
