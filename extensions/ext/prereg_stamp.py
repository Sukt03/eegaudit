"""Stamp a PREREG.json: writes PREREG_STAMP.json next to it (sha256 of the file, clock time, list of files already in the directory).
Run BEFORE any scoring. Usage: python -I -u prereg_stamp.py <dir containing PREREG.json> [extra files to hash...]"""
import hashlib, json, os, sys, time
d = sys.argv[1]
p = os.path.join(d, 'PREREG.json')
assert os.path.exists(p) and not os.path.exists(os.path.join(d, 'PREREG_STAMP.json')), 'missing PREREG or already stamped'
h = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
st = dict(prereg_sha256=h(p), stamped=time.strftime('%Y-%m-%d %H:%M:%S %Z'), unix=time.time(),
          extra_file_sha256={f: h(f) for f in sys.argv[2:]}, files_in_dir_at_stamp=sorted(os.listdir(d)),
          note='written before any scored output of this experiment (summary.json absent from files_in_dir_at_stamp)')
assert 'summary.json' not in st['files_in_dir_at_stamp']
json.dump(st, open(os.path.join(d, 'PREREG_STAMP.json'), 'w'), indent=1)
print(json.dumps(st, indent=1))
