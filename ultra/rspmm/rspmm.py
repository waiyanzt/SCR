import os
import sys

import torch.backends.openmp
from torch import autograd
from torch.utils import cpp_extension

module = sys.modules[__name__]


class RSPMMAddMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_add_mul_forward_cuda
        else:
            forward = rspmm.rspmm_add_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_add_mul_backward_cuda
        else:
            backward = rspmm.rspmm_add_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMinMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_min_mul_forward_cuda
        else:
            forward = rspmm.rspmm_min_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_min_mul_backward_cuda
        else:
            backward = rspmm.rspmm_min_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMaxMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_max_mul_forward_cuda
        else:
            forward = rspmm.rspmm_max_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_max_mul_backward_cuda
        else:
            backward = rspmm.rspmm_max_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMAddAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_add_add_forward_cuda
        else:
            forward = rspmm.rspmm_add_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_add_add_backward_cuda
        else:
            backward = rspmm.rspmm_add_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMinAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_min_add_forward_cuda
        else:
            forward = rspmm.rspmm_min_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_min_add_backward_cuda
        else:
            backward = rspmm.rspmm_min_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMaxAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_max_add_forward_cuda
        else:
            forward = rspmm.rspmm_max_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_max_add_backward_cuda
        else:
            backward = rspmm.rspmm_max_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


def generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul="mul"):
    name = "RSPMM%s%sFunction" % (sum.capitalize(), mul.capitalize())
    if not hasattr(module, name):
        raise ValueError("No generalized rspmm implementation found for summation `%s` and multiplication `%s`"
                         % (sum, mul))
    Function = getattr(module, name)

    node_in, node_out = edge_index
    key = node_in * (node_out.max() + 1) + node_out
    order = key.argsort()

    return Function.apply(edge_index[:, order], edge_type[order], edge_weight[order], relation, input)


def load_extension(name, sources, extra_cflags=None, extra_cuda_cflags=None, **kwargs):
    if extra_cflags is None:
        if sys.platform == 'win32':
            extra_cflags = ["/O2", "/openmp", "-DAT_PARALLEL_OPENMP"]
        else:
            extra_cflags = ["-Ofast"]
            if torch.backends.openmp.is_available() and not sys.platform.startswith('darwin'):
                extra_cflags += ["-fopenmp", "-DAT_PARALLEL_OPENMP"]
            else:
                extra_cflags.append("-DAT_PARALLEL_NATIVE")
    if extra_cuda_cflags is None:
        if torch.cuda.is_available():
            extra_cuda_cflags = ["-O3"]
            extra_cflags.append("-DCUDA_OP")
        else:
            new_sources = []
            for source in sources:
                if not cpp_extension._is_cuda_file(source):
                    new_sources.append(source)
            sources = new_sources

    return cpp_extension.load(name, sources, extra_cflags, extra_cuda_cflags, **kwargs)


def _try_load_cached(name):
    """Try to load a pre-compiled rspmm binary, bypassing JIT compilation.

    Search order:
    1. Local _rspmm_prebuilt.pyd — bundled pre-built binary (renamed to avoid import shadowing)
    2. PyTorch extensions cache (~/.cache/torch_extensions/...) — from previous JIT build
    """
    import importlib.util

    def _load_pyd(path):
        if sys.platform == 'win32':
            torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
            if os.path.isdir(torch_lib):
                os.add_dll_directory(torch_lib)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # 1. Check local pre-built binary (renamed to _rspmm_prebuilt to avoid import shadowing)
    local_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == 'win32':
        prebuilt = os.path.join(local_dir, "_rspmm_prebuilt.pyd")
        if os.path.isfile(prebuilt):
            return _load_pyd(prebuilt)
    else:
        prebuilt_so = os.path.join(local_dir, "_rspmm_prebuilt.so")
        if os.path.isfile(prebuilt_so):
            return _load_pyd(prebuilt_so)

    # 2. Check PyTorch JIT cache
    build_dir = cpp_extension._get_build_directory(name, verbose=False)
    for ext in ['.pyd', '.so']:
        path = os.path.join(build_dir, name + ext)
        if os.path.isfile(path):
            return _load_pyd(path)

    return None


_cached = _try_load_cached("rspmm")
if _cached is not None:
    rspmm = _cached
else:
    print("Load rspmm extension. This may take a while...")
    path = os.path.join(os.path.dirname(__file__), "source")
    rspmm = load_extension("rspmm", [os.path.join(path, "rspmm.cpp"), os.path.join(path, "rspmm.cu")])