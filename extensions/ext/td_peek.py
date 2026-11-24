"""print-only viewer of td_pilot shards (runner, 23 Nov 2026). NOT citable. Usage: td_peek.py <shard.npz> ..."""
import sys, numpy as np
for p in sys.argv[1:]:
    z = np.load(p); Y = z['Y']; M = z['META']; mn = list(z['META_names']); print(p.split('/')[-1], 'cells', len(Y), 'X1', z['X1'].shape, 'TOK', z['TOK'].shape, 'finite', bool(np.isfinite(z['X1']).all() and np.isfinite(z['TOK'].astype(float)).all()))
    for i in range(len(Y)):
        print(i, 'n', int(z['NUSE'][i]), {k: round(float(M[i, mn.index(k)]), 3) for k in ('cls', 'rho', 'a', 'b', 'f', 'w0', 'wd', 'r_n', 'r_e')}, 'YN %.3f YO %.3f YNO %.3f' % (Y[i, 0], Y[i, 2], Y[i, 4]), 'A1 %.3f A4 %.3f' % (z['BL'][i, 0], z['BL'][i, 2]))
