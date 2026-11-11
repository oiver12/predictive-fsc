"""Record the corrected FiVI source and actual Java artifact identity."""
import hashlib
import json
from pathlib import Path
import subprocess
root = Path('/opt/fivi')
identity = dict(source_repository='https://github.com/oiver12/ConstrainedPlanningToolbox',
                source_revision='d97f25187803f52b46ebcf040bf9fb8b5e832ed2',
                source_archive_sha256='322d3a51babb9d68a87a5f02696315ff2729f796bba82d2cfdc7ae459ece136e',
                artifact_sha256=hashlib.sha256((root/'fivi.jar').read_bytes()).hexdigest(),
                java=subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT,
                                             text=True).splitlines()[0], maven='3.9.9')
(root/'build.json').write_text(json.dumps(identity, indent=2)+'\n')
