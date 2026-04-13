
This is the [llama.cpp (master branch)](https://github.com/ggml-org/llama.cpp)
+ merged: https://github.com/test1111111111111112/llama-cpp-turboquant-gemma4 (feature/turboquant-kv-cache branch)

it is compiled with:

cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j96

merged with claude code.

How to run a server with gemma:

./build/bin/llama-server --host 0.0.0.0  --port 8080 -ngl 99 -fa on -ctk turbo2_tcq -ctv turbo2_tcq -m /home/marcel/.lmstudio/models/unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_S.gguf

How to run a very low memory version with Bonsai 8b:

./build/bin/llama-server --host 0.0.0.0 --port 8080 -ngl 99 -m /home/marcel/.lmstudio/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B.gguf --temp 0 --top-p 0.85 --top-k 20 -ctk turbo3_tcq -ctv turbo2_tcq
