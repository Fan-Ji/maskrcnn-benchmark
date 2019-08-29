import subprocess as sp
import os.path as osp
import os
import sys

if __name__ == '__main__':
    current_dir = osp.dirname(__file__)
    sp.call([
        f"{osp.join(sys.prefix,'bin/')}python",
        "-m",
        "grpc_tools.protoc",
        f"--python_out={current_dir}",
        f"--grpc_python_out={current_dir}",
        f"--mypy_out={current_dir}",
        f"--plugin=protoc-gen-mypy={osp.join(sys.prefix, 'bin', 'protoc-gen-mypy')}",
        f"-I.",
        "./inference_service.proto"
    ])

