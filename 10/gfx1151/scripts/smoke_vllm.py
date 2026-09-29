import os

from vllm import LLM, SamplingParams

model = os.environ.get("VLLM_SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
attention_backend = os.environ.get("VLLM_SMOKE_ATTENTION_BACKEND")
llm = LLM(
    model=model,
    dtype="float16",
    max_model_len=1024,
    max_num_seqs=1,
    gpu_memory_utilization=0.5,
    enforce_eager=True,
    attention_backend=attention_backend,
)
outputs = llm.generate(
    ["The capital of France is"],
    SamplingParams(max_tokens=8, temperature=0),
)
response = outputs[0].outputs[0].text.strip()
assert response, "Engine returned an empty completion"
print(f"Engine response: {response}")
print("vLLM engine smoke test: PASS")
