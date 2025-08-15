# LLM
from ollama import chat, ChatResponse
from google import genai

# Memory + User management
from tokens import TokenManager
from memory import Memory

# .env
from dotenv import load_dotenv
from os.path import dirname, join
from os import getenv

class LLM:
    def __init__(self) -> None:
        self.token_manager = TokenManager()
        self.memory = Memory()

        dotenv_path = join(dirname(__file__), ".env")
        load_dotenv(dotenv_path)    
        
        # Initialize variables
        self.google_token = getenv("GOOGLE_API_KEY")
        
        self.provider = getenv("PROVIDER")
        if self.provider is None:
            self.provider = "ollama"
        
        self.model = getenv("model")
        if self.model is None:
            if self.provider == "ollama":
                self.model = "gemma3:1b-it-qat"
            elif self.provider == "google":
                self.model = "gemma-3-27b-it"
        
        self.system = [
            {
                "role": "system",
                "content": """
                You are an ai assistant, your name is Nate.
                You are messages in format username: message. which is used to identify who is speaking to you.
                Be concise and follow user instructions. You can act differently based on which user speaks to you (in first impression, infer based on factors like tone and username).
                
                You will be given data in the format: Username: message. which is used to identify who is speaking to you, they may request to be identified with a different name or alias.
                
                For example (does not need exact following):
                Username: hello
                Assistant: Hello sir, how may I assist you?
                """
            },
        ]
    
    def _convert_to_google_format(self, messages: list) -> list:
        """
        Converts a list of messages from the internal format to the Google GenAI format.
        The Google API expects a list of dictionaries with 'role' and 'parts' keys.
        """
        google_format_messages = []
        for message in messages:
            # Google's API uses 'user' and 'model' for conversational roles.
            role_map = {
                "user": "user",
                "assistant": "model"
            }

            # The 'system' role is not a valid conversational role for `generate_content`.
            # This check ensures that system prompts are not included in the conversation history.
            if message['role'] in role_map:
                google_format_messages.append({
                    "role": role_map[message['role']],
                    "parts": [{"text": message['content']}]
                })
        return google_format_messages
    
    def enroll_user(self, user: str) -> str:
        """ Enrolls a user based on name, returns a token """
        try:
            return self.token_manager.generate_token(user)
        except ValueError as e:
            raise ValueError(f"User {user} is already enrolled: {e}")
    
    def delete_user(self, token: str) -> None:
        """ Deletes a token and user, with memory files too """
        
        try:
            self.token_manager.delete_token(token)
        except ValueError as e:
            print(f"Token {token} is invalid: {e}")
            raise ValueError("Token is invalid")
        
        try:
            self.memory.memory_delete(token)
        except FileNotFoundError:
            pass
    
    
    def ask_ollama(self, prompt:str, token: str) -> str: 
        """ Gets a response from the ollama provider, takes a prompt and a token to get memory file and name """
        
        messages = self.memory.memory_load(token)
        try:
            user = self.token_manager.get_user(token)
        except ValueError as e:
            print(f"Token {token} is invalid: {e}")
            raise ValueError("Token is invalid")
        
        # save user input to memory
        messages.append({
            "role": "user",
            "content": f"{user}: {prompt}"
        })
        
        full_conversation = self.system + messages
        response = chat(model=self.model, messages=full_conversation)
        
        # save assistant output to memory
        messages.append({
            "role": "assistant",
            "content": response['message']['content']
        })
        
        # save memory json to file
        self.memory.memory_append(token, messages)
        
        return response['message']['content']
    
    def ask_google(self, prompt: str, token: str) -> str:
        """ Gets a response from the google provider, takes a prompt and a token to get memory file and name """

        messages = self.memory.memory_load(token)
        try:
            user = self.token_manager.get_user(token)
        except ValueError as e:
            print(f"Token {token} is invalid: {e}")
            raise ValueError("Token is invalid")
        
        messages.append({
            "role": "user",
            "content": f"{user}: {prompt}"
        })
        
        full_conversation_for_google = self._convert_to_google_format(messages)

        try:
            # Initialize the Google GenAI client and make the API call.
            client = genai.Client(api_key=self.google_token)
            response = client.models.generate_content(
                model=self.model,
                contents=full_conversation_for_google
            )

            # The response is an object, so we must access the text attribute to get the content.
            assistant_response_content = response.text

            # Append the assistant's response to the conversation memory.
            messages.append({
                "role": "assistant",
                "content": assistant_response_content
            })

            # Save the updated conversation memory to the file.
            self.memory.memory_append(token, messages)

            # Return only the text of the assistant's response.
            return assistant_response_content

        except Exception as e:
            # Handle potential errors during the API call gracefully.
            print(f"An error occurred with the Google GenAI call: {e}")
            return "An error occurred while generating the response."
        
    def ask(self, prompt: str, token: str) -> str:
        """ Gets a response from the ollama provider, takes a prompt and a token to get memory file and name """
        
        if self.provider == "ollama":
            return self.ask_ollama(prompt, token)
        
        elif self.provider == "google":
            return self.ask_google(prompt, token)

        else:
            raise ValueError("Provider is not supported")
        
    def prompt(self, prompt: str) -> str:
        ''' Accepts an input string and returns the response as a string. '''
        
        if self.provider == "ollama":
            response: ChatResponse = chat(model=self.model, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
            ])
            return response['message']['content']
        
        elif self.provider == "google":
            client = genai.Client(api_key=self.google_token)
            response = client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text

        else:
            raise ValueError("Provider is not supported")
