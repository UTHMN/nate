from ollama import chat, ChatResponse

class llm:
    def __init__(self, model) -> None:
        self.model = model
    
    def ask(self, input: str) -> str:
        ''' Accepts an input string and returns the response as a string. '''
        
        response: ChatResponse = chat(model=self.model, messages=[
        {
            'role': 'user',
            'content': input,
        },
        ])
        return(response['message']['content'])
    
    ''' Text '''

if __name__ == "__main__":    
    model = llm("gemma3:1b-it-qat")
    
    while True:
        try:
            prompt = input("user> ")
            print(f"nate> {model.ask(prompt)}")
        except KeyboardInterrupt:
            print("\nGoodbye 👋...")
            exit(0)
        except Exception as e:
            print(f"\nUnknown exception: {e}")
            exit(1)
