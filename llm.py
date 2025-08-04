# LLM
from ollama import chat, ChatResponse

# Memory
from json import loads, dumps
from os.path import exists

# .env
from dotenv import load_dotenv
from os.path import dirname, join
from os import getenv

class LLM:
    def __init__(self) -> None:
        dotenv_path = join(dirname(__file__), ".env")
        load_dotenv(dotenv_path)
        
        # Initialize variables
        self.model = getenv("model")
        if self.model is None:
            self.model = "gemma3:1b-it-qat"
        
        self.messages = self.memory_load()
        self.system = [
            {
                "role": "system",
                "content": """
                You are an ai assistant, your name is Nate.
                You are messages in format username: message. which is used to identify who is speaking to you.
                Be concise and follow user instructions. You can act differently based on which user speaks to you (in first impression, infer based on factors like tone and username).
                
                For example (does not need exact following):
                User: hello
                Assistant: Hello sir, how may I assist you?
                """
            },
        ]
    
    def ask(self, prompt: str, user: str) -> str:
        ''' Prompts an AI based on memory '''
        
        # save user input to memory
        self.messages.append({
            "role": "user",
            "content": f"{user}: {prompt}"
        })
        
        full_conversation = self.system + self.messages
        response = chat(model=self.model, messages=full_conversation)
        
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