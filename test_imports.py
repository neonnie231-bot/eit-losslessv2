import importlib
import pytest

MODULES = [
    "eit3_pack_int4_kernel",
    "eit3_kernels",
    "eit3_engine",
    "eit3_controller",
    "eit3_toy_decoder",
]

@pytest.mark.parametrize("modname", MODULES)
def test_module_imports(modname):
    importlib.import_module(modname)

def test_optional_entrypoints():
    for m in MODULES:
        spec = importlib.util.find_spec(m)
        if spec is None:
            continue
        mobj = importlib.import_module(m)
        for attr in ("example", "demo", "run", "main"):
            if hasattr(mobj, attr):
                assert callable(getattr(mobj, attr))
