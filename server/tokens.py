import uuid
import json

# Generate and correlate a token for a user, support user enrollment and if a user is already enrolled.

class TokenManager:
    def __init__(self) -> None:
        # Load tokens from file
        try:
            with open("tokens.json", "r") as f:
                self.tokens = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.tokens = {}
            
    def is_enrolled(self, user: str) -> bool:
        return user.lower() in self.tokens.values()
    
    def token_exists(self, token: str) -> bool:
        return token in self.tokens
    
    def generate_token(self, user: str) -> str:
        user = user.lower()
        if self.is_enrolled(user):
            raise ValueError("User is already enrolled")
        # Generate a new token for the user
        token = str(uuid.uuid4())
        
        # Add the token to the tokens dictionary
        self.tokens[token] = user
        
        # Save the tokens dictionary to the file
        with open("tokens.json", "w") as f:
            json.dump(self.tokens, f, indent=4)
        
        return token
    
    def get_user(self, token: str) -> str:
        # Check if the token is valid
        if token not in self.tokens:
            raise ValueError("Token is invalid")
        
        # Return the user associated with the token
        return self.tokens[token]
    
    def get_token(self, user: str) -> str:
        user = user.lower()
        # Check if the user is enrolled
        if not self.is_enrolled(user):
            raise ValueError("User is not enrolled")
        
        # Return the token associated with the user
        return list(self.tokens.keys())[list(self.tokens.values()).index(user)]
    
    def delete_token(self, token: str) -> None:
        # Check if the token is valid
        if token not in self.tokens:
            raise ValueError("Token is invalid")
        
        # Delete the token from the tokens dictionary
        del self.tokens[token]
        
        # Save the tokens dictionary to the file
        with open("tokens.json", "w") as f:
            json.dump(self.tokens, f, indent=4)
