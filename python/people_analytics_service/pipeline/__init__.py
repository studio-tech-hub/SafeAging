"""P2-1: Pipeline decomposition of the /infer detection pipeline.

`_run_person_detection_pipeline` in api.py used to be one ~900-line function
mixing preprocessing, YOLO inference, tracking, fall detection, face
recognition, zone checks, and persistence concerns. It is now a thin
orchestrator (see api.py) that builds an `InferContext` and runs it through
an ordered list of `Stage` objects defined under `pipeline/stages/`.

Behaviour is unchanged from before this refactor — this is a mechanical
Extract Class + Pipeline-pattern change, gated by the characterization
("golden") test at python/tests/unit/test_infer_golden.py, which must stay
green, unmodified, after every single stage extraction.

DecodeStage is intentionally NOT one of the Stage objects here: JSON+base64
(`/infer`) and multipart (`/infer/binary`) use different, transport-specific
decoders (`_decode_current_transport_image` / `_decode_binary_transport_image`
in api.py, extracted back in P1-4) and both already hand off a plain decoded
BGR frame + transport metadata to this pipeline — decode is effectively
"stage 0", already isolated at the route-handler level before an
InferContext is even constructed.
"""
from .context import InferContext
from .stages import PIPELINE_STAGES

__all__ = ["InferContext", "PIPELINE_STAGES"]
