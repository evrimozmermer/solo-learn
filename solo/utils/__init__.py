# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from solo.utils import (
    checkpointer,
    classification_dataloader,
    knn,
    lars,
    metrics,
    misc,
    momentum,
    pretrain_dataloader,
    sinkhorn_knopp,
)

__all__ = [
    "classification_dataloader",
    "pretrain_dataloader",
    "checkpointer",
    "knn",
    "misc",
    "lars",
    "metrics",
    "momentum",
    "sinkhorn_knopp",
]

try:
    from solo.utils import dali_dataloader  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("dali_dataloader")

try:
    from solo.utils import auto_umap  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("auto_umap")
