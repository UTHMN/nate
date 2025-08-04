import typer
import requests
import os
import sys

try:
    import readline  # for command history and line editing on Unix
except ImportError:
    readline = None

app = typer.Typer()

def clear_screen():
    # Cross-platform clear
    if sys.platform == "win32":
        os.system("cls")
    else:
        os.system("clear")

@app.command()
def chat(name: str = typer.Argument(..., help="Your username")):
    typer.echo(f"Welcome, {name}! Type 'exit' to quit, 'clear' to clear the screen.")

    if readline:
        # Enable simple history navigation
        readline.parse_and_bind("tab: complete")
        # Optional: Save/load history to a file (add if you want persistent history)
    
    while True:
        try:
            message = input("> ")
        except EOFError:
            # Handles Ctrl+D on Unix (end of input)
            print("\nGoodbye👋...")
            raise typer.Exit()

        message = message.strip()

        if not message:
            continue
        if message.lower() == "exit":
            print("Exiting... Bye!")
            raise typer.Exit()
        elif message.lower() == "clear":
            clear_screen()
            continue
        
        try:
            response = requests.post("http://localhost:8000/ask", json={"prompt": message, "user": name})
            response.raise_for_status()  # raise error on bad status
            json_resp = response.json()
            typer.echo(json_resp.get("message", "[No message in response]"))
        except requests.RequestException as e:
            typer.echo(f"Request error: {e}")
        except Exception as e:
            typer.echo(f"Unexpected error: {e}")

if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        print("\nGoodbye👋...")
