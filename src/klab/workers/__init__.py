"""Independent worker contracts and finite stdio transport."""

from klab.workers.models import (
    ArtifactIdentity,
    AttemptReceipt,
    BundleManifest,
    ContinuationSpec,
    ExperimentSpec,
    MatrixMetadata,
    SchedulingRequirements,
    Submission,
    WorkerCapabilities,
    WorkerDefinition,
    current_required_versions,
    validate_required_versions,
)
from klab.workers.transport import (
    ProtocolError,
    ProtocolReply,
    RemoteProtocolError,
    WorkerBusyError,
    call_worker,
    read_frame,
    write_frame,
)

__all__ = [
    "ArtifactIdentity",
    "AttemptReceipt",
    "BundleManifest",
    "ContinuationSpec",
    "ExperimentSpec",
    "MatrixMetadata",
    "ProtocolError",
    "ProtocolReply",
    "RemoteProtocolError",
    "SchedulingRequirements",
    "Submission",
    "WorkerBusyError",
    "WorkerCapabilities",
    "WorkerDefinition",
    "call_worker",
    "current_required_versions",
    "read_frame",
    "validate_required_versions",
    "write_frame",
]
