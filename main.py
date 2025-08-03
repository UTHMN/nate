# LLM
from ollama import chat, ChatResponse

# Memory
from json import loads, dumps
from os.path import exists

# .env
from dotenv import load_dotenv
from os.path import dirname, join
from os import getenv

dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)

class LLM:
    def __init__(self) -> None:
        # Initialize variables
        self.model = getenv("model")
        self.messages = self.memory_load()
    
    def ask(self, prompt: str) -> str:
        ''' Prompts an AI based on memory '''
        
        # save user input to memory
        self.messages.append({
            "role": "user",
            "content": prompt
        })
        
        response = chat(model=self.model, messages=self.messages)
        
        # save assistant output to memory
        self.messages.append({
            "role": "assistant",
            "content": response['message']['content']
        })
        
        # save memory json to file
        self.memory_append()
        
        return response['message']['content']
    
    def prompt(self, prompt: str) -> str:
        ''' Accepts an input string and returns the response as a string. '''
        
        response = chat(model=self.model, messages=[
            {
                "role": "user",
                "content": prompt
            }
        ])
        
        return(response['message']['content'])
    
    def memory_append(self) -> None:
        ''' Appends to memory.json file '''
        
        path = join(dirname(__file__), "messages.json")
        
        if not exists(path): return
        
        with open(path, "w") as f:
            f.write(dumps(self.messages, indent=4))
            
    def memory_load(self) -> str:
        ''' Loads memory from the memory.json file '''
        
        path = join(dirname(__file__), "messages.json")
        
        if not exists(path):
            with open(path, "w") as f:
                f.write("[]")
        
            return []
        
        with open(path, "r") as f:
            data = loads(f.read())
            return data
    
    # def memory_compress(self)
    # Uses self.prompt to summarise all memory until previous call

if __name__ == "__main__":
    model = LLM()
    
    while True:
        try:
            prompt = input("user> ")
            print(f"nate> {model.ask(prompt)}")
        
        except KeyboardInterrupt:
            print("Goodbye 👋...")
            exit(0)
