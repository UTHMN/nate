import json
from os import makedirs, remove
from os.path import exists, join

class Memory:
    def __init__(self, path: str | None = "memories") -> None:
        self.path = path
        
        if not exists(path):
            makedirs(path)
    
    def memory_load(self, token: str) -> list:
        try:
            with open(join(self.path, f"{token}.json"), "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []
         
    def memory_append(self, token: str, memory: list) -> None:
        with open(join(self.path, f"{token}.json"), "w") as f:
            json.dump(memory, f, indent=4)

    def memory_delete(self, token: str) -> None:
        try:
            remove(join(self.path, f"{token}.json"))
        except FileNotFoundError:
            pass
