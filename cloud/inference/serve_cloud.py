import sys
import runpy

from transformers import PretrainedConfig, AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from vllm import ModelRegistry

from cloud.inference.vllm_llama_cloud import LlamaCloudModel


# Custom config
class CLoudConfig(PretrainedConfig):
    model_type = "cloud"

    def __init__(self, feedback_method="vanilla", base_model_name_or_path="meta-llama/Meta-Llama-3-8B", **kwargs):
        
        assert feedback_method in ["vanilla", "teacher"]

        self.feedback_method = feedback_method
        self.base_model_name_or_path = base_model_name_or_path

        original_config = AutoConfig.from_pretrained(base_model_name_or_path)
        # Add attributes from original_config that are not present in CLoudConfig
        for key, value in original_config.__dict__.items():
            if key not in self.__dict__ and key not in kwargs:
                kwargs[key] = value

        super().__init__(**kwargs)

CONFIG_MAPPING.register(CLoudConfig.model_type, CLoudConfig)

# Register cloud models with vllm
ModelRegistry.register_model("FeedbackRewardModel", LlamaCloudModel) # Change to CloudRewardModel

if __name__ == "__main__":

    sys.argv = sys.argv + ["--enforce-eager"]
    runpy.run_module(
        'vllm.entrypoints.openai.api_server',
        run_name='__main__',
    )
