import os
from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams


class InferlessPythonModel:
    def initialize(self):        
        nfs_volume = os.getenv("NFS_VOLUME")
        if os.path.exists(nfs_volume + "/tinyllama-1.1b-chat-v1.0.Q4_0.gguf") == False :
            cache_file = hf_hub_download(
                                repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                                filename="tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
                                local_dir=nfs_volume)
        self.llm = LLM(model=f"{nfs_volume}/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
                  tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                  gpu_memory_utilization=0.95)
        
    def infer(self, inputs):
        prompt = inputs["prompt"]
        system_prompt = inputs.get("system_prompt","You are a friendly bot.")
        temperature = inputs.get("temperature",0.7)
        top_p = inputs.get("top_p",0.1)
        top_k = int(inputs.get("top_k",40))
        repetition_penalty = inputs.get("repetition_penalty",1.18)
        max_tokens = inputs.get("max_tokens",256)
        
        CHAT_TEMPLATE = "<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"  # noqa: E501

        prompts = [CHAT_TEMPLATE.format(system_prompt=system_prompt, prompt=prompt)]
        sampling_params = SamplingParams(temperature=temperature,top_p=top_p,repetition_penalty=repetition_penalty,
                                 top_k=top_k,max_tokens=max_tokens)
        result = self.llm.generate(prompts, sampling_params)
        result_output = [output.outputs[0].text for output in result]

        return {'result': result_output[0]}        
    def finalize(self):
        self.llm = None
