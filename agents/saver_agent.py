# pipeline/agents/saver_agent.py

class SaverAgent:

    def __init__(self):
        pass

    def save_raw(self, content: str, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
