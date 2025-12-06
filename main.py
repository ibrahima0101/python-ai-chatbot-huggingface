

from chatbot import get_interface

if __name__ == "__main__":
    _ui = get_interface()
    _ui.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)