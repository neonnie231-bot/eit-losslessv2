import sys
import importlib

def print_env():
    print("PYTHON:", sys.version.split()[0])
    try:
        import torch
        print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
    except Exception as e:
        print("torch import error:", repr(e))

    try:
        import triton
        print("triton:", triton.__version__)
    except Exception as e:
        print("triton import error or not installed:", repr(e))

def try_call(m, names):
    for n in names:
        if hasattr(m, n):
            attr = getattr(m, n)
            if callable(attr):
                try:
                    print(f"Running {m.__name__}.{n}()")
                    attr()
                except Exception as e:
                    print(f"{m.__name__}.{n}() raised:", repr(e))
                return True
    return False

def main():
    print_env()
    targets = [
        "eit3_pack_int4_kernel",
        "eit3_kernels",
        "eit3_engine",
        "eit3_controller",
        "eit3_toy_decoder",
    ]
    for t in targets:
        try:
            m = importlib.import_module(t)
            print(f"Imported {t}")
            if not try_call(m, ("example", "demo", "run", "main")):
                print(f"No runnable example() found in {t}; module OK.")
        except Exception as e:
            print(f"Failed import {t}: {repr(e)}")

if __name__ == "__main__":
    main()
