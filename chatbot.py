

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
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

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
    
    # append user prompt to the chat history
    chat_history.append(f"User: {prompt}")
    print("chat_history (after user prompt):", chat_history)

    # the input text considering the chat history
    model_input_text = "\n".join(chat_history)
    print("model_input_text:", model_input_text)

    # tokenize the model_input_text (encode_plus used as we give new prompt and chat history together)
    inputs = tokenizer(model_input_text, return_tensors="pt")
    # print(inputs)

    # generate model response
    outputs = model.generate(**inputs)
    # print(outputs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("response:", response)

    # append model response to the chat history
    chat_history.append(f"Model: {response}")
    print("chat_history (after model response):", chat_history)

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

