
This is the [llama.cpp (master branch)](https://github.com/ggml-org/llama.cpp)
+ merged: https://github.com/test1111111111111112/llama-cpp-turboquant-gemma4 (feature/turboquant-kv-cache branch)

it is compiled with:

docker run -it --gpus all -p 8080:8080 -v ~/mh/tests:/mnt/tests -v "$HOME/.cache/huggingface:/root/.cache/huggingface" docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04 bash
apt update && apt install -y cmake git libssl-dev

cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_CUB_3DOT2=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j96

merged with claude code.

How to run a server with gemma:

./build/bin/llama-server --host 0.0.0.0 --port 8080 -ngl 99 -fa on -ctk q8_0 -ctv turbo4 --jinja --parallel 1 -hf unsloth/gemma-4-E4B-it-GGUF:Q4_K_S --kv-unified

./build/bin/llama-server \
  --host 0.0.0.0 --port 8080 \
  -hf unsloth/gemma-4-E4B-it-GGUF:Q4_K_S \
  -ngl 99 \
  -fa on \
  -ctk q8_0 -ctv turbo4 \
  -c 32768 \
  --parallel 1 \
  -t 4 \
  --jinja


./build/bin/llama-server --host 0.0.0.0 --port 8080 -ngl 99 -fa on -ctk q8_0 -ctv turbo4 --jinja --no-mmap --parallel 12 -hf unsloth/gemma-4-E4B-it-GGUF:Q4_K_S --kv-unified

./build/bin/llama-server \
  --host 0.0.0.0 --port 8080 \
  -hf unsloth/gemma-4-E4B-it-GGUF:Q4_K_S \
  -ngl 99 \
  -fa on \
  -ctk q8_0 -ctv turbo4 \
  -c 65000 \
  --parallel 1 \
  -t 4 \
  --jinja
  --kv-unified

./build/bin/llama-server \
  --host 0.0.0.0 --port 8080 \
  -hf unsloth/gemma-4-31B-it-GGUF:UD-IQ2_XXS \
  -ngl 99 \
  -fa on \
  -ctk q8_0 -ctv turbo4 \
  -c 65000 \
  --parallel 1 \
  -t 4 \
  --jinja
  --kv-unified

./build/bin/llama-server \
  --host 0.0.0.0 --port 8080 \
  -hf unsloth/gemma-4-E4B-it-GGUF:Q4_K_S \
  -ngl 99 \
  -fa on \
  -ctk q8_0 -ctv turbo4 \
  -c 65000 \
  --parallel 1 \
  -t 4 \
  --jinja
  --kv-unified