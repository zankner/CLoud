from argparse import ArgumentParser

import gradio as gr
from datasets import load_dataset

from cloud.inference.api import CLoudAPI

data = load_dataset("allenai/reward-bench")["filtered"].shuffle(seed=42)

def create_gradio_interface(cloud_api, classic_api, cloud_model_name, classic_model_name):
    def process_input(prompt, chosen_response, rejected_response):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_cloud_chosen = executor.submit(cloud_api.get_reward, prompt, chosen_response)
            future_cloud_rejected = executor.submit(cloud_api.get_reward, prompt, rejected_response)
            future_classic_chosen = executor.submit(classic_api.get_reward, prompt, chosen_response)
            future_classic_rejected = executor.submit(classic_api.get_reward, prompt, rejected_response)

        cloud_chosen_critique, cloud_chosen_reward = future_cloud_chosen.result()
        cloud_rejected_critique, cloud_rejected_reward = future_cloud_rejected.result()
        _, classic_chosen_reward = future_classic_chosen.result()
        _, classic_rejected_reward = future_classic_rejected.result()

        return (
            cloud_chosen_critique,
            cloud_chosen_reward,
            classic_chosen_reward,
            cloud_rejected_critique,
            cloud_rejected_reward,
            classic_rejected_reward,
        )

    def load_example(index):
        example = data[index]
        return (
            example["prompt"],
            example["chosen"],
            example["rejected"],
        )
    
    # Load initial example
    initial_prompt, initial_chosen, initial_rejected = load_example(0)

    with gr.Blocks() as demo:
        gr.Markdown("# CLoud vs Classic Reward Bench Demo")
        gr.Markdown("The prompts, chosen, and rejected responses are sampled from the [reward-bench dataset](https://github.com/allenai/reward-bench). For each input, we sample the reward according to both the CLoud and Classic Reward Models.")
        
        # Add information about the models being used
        gr.Markdown(f"""
        ## Models Used
        - CLoud Model: {cloud_model_name}
        - Classic Model: {classic_model_name}
        """)
        
        prompt = gr.Textbox(lines=3, label="Prompt", value=initial_prompt)
        with gr.Row():
            with gr.Column():
                chosen_response = gr.Textbox(lines=10, label="Chosen Response", value=initial_chosen)
            with gr.Column():
                rejected_response = gr.Textbox(lines=10, label="Rejected Response", value=initial_rejected)

        with gr.Row():
            prev_btn = gr.Button("Previous")
            next_btn = gr.Button("Next")
        
        submit_btn = gr.Button("Submit")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Chosen Response Analysis")
                cloud_chosen_critique = gr.Textbox(lines=10, label="Cloud Chosen Critique", interactive=False)
                cloud_chosen_reward = gr.Number(label="Cloud Chosen Reward", interactive=False)
                classic_chosen_reward = gr.Number(label="Classic Chosen Reward", interactive=False)
            
            with gr.Column():
                gr.Markdown("### Rejected Response Analysis")
                cloud_rejected_critique = gr.Textbox(lines=10, label="Cloud Rejected Critique", interactive=False)
                cloud_rejected_reward = gr.Number(label="Cloud Rejected Reward", interactive=False)
                classic_rejected_reward = gr.Number(label="Classic Rejected Reward", interactive=False)
        
        
        example_index = gr.State(value=0)

        def update_example(index, direction):
            new_index = (index + direction) % len(data)
            return (new_index,) + load_example(new_index)
        
        prev_btn.click(
            update_example,
            inputs=[example_index, gr.Number(value=-1, visible=False)],
            outputs=[example_index, prompt, chosen_response, rejected_response]
        )
        
        next_btn.click(
            update_example,
            inputs=[example_index, gr.Number(value=1, visible=False)],
            outputs=[example_index, prompt, chosen_response, rejected_response]
        )
        
        submit_btn.click(
            process_input,
            inputs=[prompt, chosen_response, rejected_response],
            outputs=[cloud_chosen_critique, cloud_chosen_reward, classic_chosen_reward,
                     cloud_rejected_critique, cloud_rejected_reward, classic_rejected_reward]
        )
        
    
    return demo

def main(args):
    cloud_api = CLoudAPI(args.cloud_model, hosted=True, server_url=f"http://localhost:{args.cloud_port}")
    classic_api = CLoudAPI(args.classic_model, hosted=True, server_url=f"http://localhost:{args.classic_port}")
    demo = create_gradio_interface(cloud_api, classic_api, args.cloud_model, args.classic_model)
    return demo

# if __name__ == "__main__":
parser = ArgumentParser()
parser.add_argument("--cloud-model", type=str, default="ankner/Llama3-8B-CLoud-RM")
parser.add_argument("--cloud-port", type=int, default=8000)
parser.add_argument("--classic-model", type=str, default="ankner/Llama3-8B-Classic-RM")
parser.add_argument("--classic-port", type=int, default=8001)
args = parser.parse_args()
demo = main(args)
demo.launch(share=True)
