


## Openfrontier

### Instalation
```bash
uv venv --python 3.12 .venv_open_fron
source .venv_open_fron/bin/activate
export CMAKE_POLICY_VERSION_MINIMUM=3.5
cd external/OpenFrontier/
uv pip install -r requirements.txt
bash scripts/download_weights.sh

git submodule update --init third_party/UniK3D
uv pip install -e third_party/UniK3D
uv pip install "setuptools<81"
```