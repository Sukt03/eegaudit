"""Loading third-party torch files safely.
1) torch.load(weights_only=True), allow-listing ONLY numpy globals as the unpickler reports them.
2) If the weights-only unpickler cannot parse the stream (e.g. 'Unsupported operand'), scan EVERY global referenced in the pickle
   (pickletools opcodes GLOBAL / STACK_GLOBAL, including the string pushes that feed STACK_GLOBAL) and load with weights_only=False ONLY if
   every global is in numpy.*, torch._utils.*, torch.*Storage, collections.OrderedDict or builtins set/frozenset/slice. Otherwise raise."""
import importlib, io, pickletools, re, zipfile, torch

OK_PREFIX = ('numpy.', 'numpy._core', 'torch._utils.', 'collections.OrderedDict', 'builtins.set', 'builtins.frozenset', 'builtins.slice')


def scan_globals(p):
    """Every global the pickle can import. Tracks the memo so STACK_GLOBAL operands fetched with BINGET are resolved."""
    z = zipfile.ZipFile(p); name = [n for n in z.namelist() if n.endswith('data.pkl')][0]
    data = z.read(name); found = set(); pushed = []; memo = {}
    for op, arg, _ in pickletools.genops(io.BytesIO(data)):
        if op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE', 'BINUNICODE8', 'SHORT_BINSTRING', 'BINSTRING', 'STRING'):
            pushed.append(arg)
        elif op.name == 'MEMOIZE':
            memo[len(memo)] = pushed[-1] if pushed else None
        elif op.name in ('BINPUT', 'LONG_BINPUT', 'PUT'):
            memo[arg] = pushed[-1] if pushed else None
        elif op.name in ('BINGET', 'LONG_BINGET', 'GET'):
            pushed.append(memo.get(arg))
        elif op.name == 'GLOBAL':
            found.add(arg.replace(' ', '.')); pushed.append(None)
        elif op.name == 'STACK_GLOBAL':
            found.add(f'{pushed[-2]}.{pushed[-1]}'); pushed.append(None)
        else:
            pushed.append(None)
    return found


def safe_load(p):
    allowed = []
    for _ in range(40):
        try:
            with torch.serialization.safe_globals(allowed):
                return torch.load(p, map_location='cpu', weights_only=True)
        except Exception as e:
            m = re.search(r'Unsupported global: GLOBAL ([\w\.]+)', str(e)) or re.search(r"but got <class '([\w\.]+)'>", str(e))
            if m and m.group(1).startswith('numpy'):
                mod, name = m.group(1).rsplit('.', 1); allowed.append(getattr(importlib.import_module(mod), name)); continue
            if 'Unsupported operand' not in str(e):
                raise
            g = scan_globals(p)
            bad = [x for x in g if not (x.startswith(OK_PREFIX) or (x.startswith('torch.') and x.endswith('Storage')))]
            if bad:
                raise RuntimeError(f'refusing {p}: non-allow-listed globals {bad}')
            print(f'[safe_load] {p.rsplit("/", 1)[-1]}: weights-only parser unsupported; all {len(g)} globals allow-listed {sorted(g)}; loading', flush=True)
            return torch.load(p, map_location='cpu', weights_only=False)
    raise RuntimeError('too many globals')
