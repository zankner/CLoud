import gradio as gr
from transformers import AutoTokenizer

from cloud.model import CLoudRewardModel

MODEL_NAME = "ankner/Llama3-8B-CLoud-RM"

def load_model_and_tokenizer():
    model = CLoudRewardModel.from_pretrained(MODEL_NAME, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    return model, tokenizer

def create_gradio_interface(model, tokenizer):
    def process_input(prompt, response):
        rewards, critiques = model.predict_reward([prompt], [response], tokenizer)
        return critiques[0], f"{rewards[0]:.4f}"

    return gr.Interface(
        fn=process_input,
        inputs=[
            gr.Textbox(lines=3, label="Prompt"),
            gr.Textbox(lines=10, label="Response")
        ],
        outputs=[
            gr.Textbox(lines=10, label="Critique", interactive=False),
            gr.Textbox(label="Reward", interactive=False)
        ],
        title="CLoud Reward Model Demo",
        description="Enter a prompt and a response to generate a critique and reward using the CLoud Reward Model.",
        allow_flagging="never"
    )

def main():
    model, tokenizer = load_model_and_tokenizer()
    demo = create_gradio_interface(model, tokenizer)
    demo.launch(share=True)

if __name__ == "__main__":
    main()