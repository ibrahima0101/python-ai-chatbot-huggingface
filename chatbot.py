

# import libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import gradio as gr

# variable to keep chat history
chat_history = []
model_cache = {}

def load_model(model_name):
    if model_name in model_cache:
        return model_cache[model_name]
    
    if model_name in ["google/flan-t5-small", "google/flan-t5-base","facebook/blenderbot-400M-distill"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            low_cpu_mem_usage=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            low_cpu_mem_usage=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_cache[model_name] = (model, tokenizer)

    return model, tokenizer

# loading the mode and tokenizer
def select_model(prompt, model_name):
    model, tokenizer = load_model(model_name)

    response = generate_response(prompt, model, tokenizer)

    return response

# function to generate response from the model
def generate_response(prompt, model, tokenizer):
    
    conversation = ""
    for u, a in chat_history:
        conversation += f"User: {u}\nAssistant: {a}\n"
    conversation += f"User: {prompt}\nAssistant: "

    # tokenize the model_input_text (encode_plus used as we give new prompt and chat history together)
    inputs = tokenizer(conversation, 
                       return_tensors="pt", 
                       truncation=True, 
                       max_length=256)

    # generate model response
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("response:", response)

    # extract only the assistant reply
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()

    chat_history.append((prompt, response))
    return response 

# Note: use below to explore vocabulary files for pretrained models
# tokenizer.pretrained_vocab_files_map

def get_interface():
    return gr.Interface(
        fn=select_model,
        inputs=[
            gr.Textbox(label="Message", placeholder="Type your message here..."),
            gr.Dropdown(
            choices=[
                "Qwen/Qwen2-1.5B-Instruct",
                "microsoft/phi-2",
                "facebook/blenderbot-400M-distill",
                "distilgpt2",
                "gpt2",
                "facebook/opt-1.3b",
                "google/flan-t5-small",
                "google/flan-t5-base",
                "tiiuae/falcon-7b-instruct",
            ],
            label="Select Model",
            value="Qwen/Qwen2-1.5B-Instruct")],
        outputs=gr.Textbox(label="Conversation"),
        title="Hi, I am ChattyBot!",
        description="I am an AI chatbot built to assist you with your queries. \n Waiting for your message... :)"
        )

