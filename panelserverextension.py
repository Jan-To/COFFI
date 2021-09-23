from subprocess import Popen

def load_jupyter_server_extension(nbapp):
    """serve the embedding.ipynb directory with bokeh server"""
    Popen(["panel", "serve", "embedding_tool2.1.ipynb", "--allow-websocket-origin=*"])
