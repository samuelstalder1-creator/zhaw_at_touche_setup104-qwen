FROM webis/touche25-ad-detection:0.0.1

ADD predict.py /predict.py
ADD requirements.txt /requirements.txt
ADD model /model

RUN python3 - <<'PY'
from pathlib import Path

source = Path("/requirements.txt")
target = Path("/requirements-docker.txt")

lines = []
for raw_line in source.read_text(encoding="utf-8").splitlines():
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("torch"):
        continue
    lines.append(raw_line)

target.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

# Reuse the torch already shipped in the base image to keep the layer size down.
RUN pip3 install --no-cache-dir -r /requirements-docker.txt
RUN pip3 uninstall -y torchvision

ARG EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
ARG QWEN_MODEL=Qwen/Qwen2.5-1.5B-Instruct

RUN python3 - <<PY
import json
import pickle
from pathlib import Path
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

embedding_model = "${EMBEDDING_MODEL}"
qwen_model = "${QWEN_MODEL}"

AutoTokenizer.from_pretrained(embedding_model)
AutoModel.from_pretrained(embedding_model)
AutoTokenizer.from_pretrained(qwen_model)
AutoModelForCausalLM.from_pretrained(qwen_model)

json.loads(Path("/model/embedding_state.json").read_text(encoding="utf-8"))
with Path("/model/embedding_lr_classifier.pkl").open("rb") as handle:
    pickle.load(handle)
PY

ENTRYPOINT ["python3", "/predict.py"]
