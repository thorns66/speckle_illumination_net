"""Read-only adapter to the validated, exact sparse light-field operator."""
import sys
from .data import ROOT, SPARSE


def operator(device):
    # The existing implementation is imported read-only, including its custom adjoint.
    # The comparison launchers disable bytecode writes.
    sys.path.append(str(ROOT))
    from tools.mixed_resolution_lfm import SparseOnlyLFM
    return SparseOnlyLFM(SPARSE,device)
