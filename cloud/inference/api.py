import subprocess
import time
import requests

from openai import OpenAI
from transformers import AutoTokenizer 

from cloud.model import COT_PROMPT, CLoudRewardModelConfig

DEFAULT_SAMPLING_KWARGS = {
    "max_tokens": 2048,
    "temperature": 1e-5,
}

class CLoudAPI():

    """
    API for interacting with the Cloud Reward Model.
    Supports both vllm non-hosted models and hosted models w/ openai API.
    """

    def __init__(self, model:str, hosted:bool=False, tensor_parallel_size:int=1, server_api_key="EMPTY", server_url="http://localhost:8000"):
        self.model = model
        self.server_url = server_url
        self.hosted = hosted

        if not hosted:
            # Start the server subprocess
            assert "localhost" in server_url, "Server URL must be localhost for non-hosted models"
            port = server_url.split(":")[-1]
            self.server_process = subprocess.Popen([
                "python", "cloud/inference/serve_cloud.py",
                "--model", model,
                "--tensor-parallel-size", str(tensor_parallel_size),
                "--port", port,
                "--disable-log-stats",
                "--disable-log-requests",
            ], stdout=subprocess.DEVNULL)
            server_api_key = "EMPTY"
            
            # Wait for the server to start
            self._wait_for_server()

        server_url += "/v1"
        self.client = OpenAI(api_key=server_api_key, base_url=server_url)

        self.is_cloud = CLoudRewardModelConfig.from_pretrained(model).feedback_method == "teacher"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.eot_token_id = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
        

    def _wait_for_server(self, timeout=600, check_interval=1):
        start_time = time.time()
        print("Waiting for server to start...")
        while time.time() - start_time < timeout:
            try:
                requests.get(f"http://localhost:8000/health")
                print("Server started")
                return
            except requests.RequestException:
                time.sleep(check_interval)
        raise TimeoutError("Server failed to start within the specified timeout.")
    
    def stop_server(self):
        if not self.hosted and hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()
            delattr(self, 'server_process')

    def __del__(self):
        self.stop_server()

    def _get_critique_and_reward(
        self,
        input_prefix: str,
        **kwargs
    ):
        res = self.client.completions.create(
            prompt=input_prefix,
            model=self.model,
            logprobs=1,
            extra_body={"stop_token_ids": [128002, self.tokenizer.eos_token_id], "repetition_penalty": 1.01},
            **kwargs
        ) 
        critique = res.choices[0].text.replace("<|eot_id|>", "").strip()
        reward = res.choices[0].logprobs.token_logprobs[-1]
        stop_reason = res.choices[0].finish_reason
        return critique, reward, stop_reason
        

    def get_reward(
        self,
        user_prompt: str,
        assistant_response: str,
        critique_prompt=COT_PROMPT,
        **kwargs
    ):

        if "temperature" in kwargs:
            assert kwargs["temperature"] != 0.0, "Bug if temperature is 0.0. Use small value like 1e-5 instead."

        if "extra_body" in kwargs:
            assert "stop_token_ids" not in kwargs["extra_body"], "Stop token is fixed"
        
        for k, v in DEFAULT_SAMPLING_KWARGS.items():
            if k not in kwargs:
                kwargs[k] = v

        input_prefix = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
            tokenize=False
        )

        if self.is_cloud:
            critique_prefix = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": critique_prompt}],
                tokenize=False
            )
            critique_prefix = critique_prefix.replace(self.tokenizer.decode([self.tokenizer.bos_token_id]), "")
            critique_prefix = critique_prefix.replace(self.tokenizer.decode([self.tokenizer.eos_token_id]), "")
            critique_prefix = critique_prefix.replace("<|eot_id|>", "") # Currently hard coded for llama3 chat template

            input_prefix += critique_prefix

        critique, reward, stop_reason = self._get_critique_and_reward(input_prefix, **kwargs)

        # Hack we have to do for now to make sure scoring on  if generation cut short
        # Fix eos token here
        if stop_reason == "length":
            input_prefix += ""
            new_critique, reward, _ = self._get_critique_and_reward(input_prefix, **kwargs)
            critique += new_critique

        
        return critique, reward

if __name__ == "__main__":
    api = CLoudAPI(model="ankner/Llama3-8B-CLoud-RM", tensor_parallel_size=1, hosted=False)
    try:
        critique, reward = api.get_reward(
            user_prompt="Write me a story",
            assistant_response="No I don't want to do that.",
            max_tokens=2048,
            temperature=1e-5
        )
        print("Critique:", critique)
        print("Reward:", reward)
    finally:
        api.stop_server()
