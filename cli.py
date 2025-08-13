# cli.py
import typer
import requests
import json
from pathlib import Path
from typing_extensions import Annotated
from rich.console import Console
from rich.prompt import Prompt, Confirm

# Initialize Rich Console for pretty output
console = Console()

CONFIG_FILE = Path("config.json")

class AppConfig:
    def __init__(self, server_url: str, server_port: int, token: str | None = None):
        self.server_url = server_url
        self.server_port = server_port
        self.token = token

    @property
    def base_url(self):
        return f"{self.server_url}:{self.server_port}"

    def to_dict(self):
        return {
            "server_url": self.server_url,
            "server_port": self.server_port,
            "token": self.token
        }

def load_config() -> AppConfig | None:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config_data = json.load(f)
            return AppConfig(
                server_url=config_data["server_url"],
                server_port=config_data["server_port"],
                token=config_data.get("token")
            )
        except (json.JSONDecodeError, KeyError) as e:
            console.print(f"[bold red]Error reading config file:[/bold red] {e}")
            return None
    return None

def save_config(config: AppConfig):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config.to_dict(), f, indent=4)
    console.print("[bold green]Configuration saved![/bold green]")

def setup_interactive_config() -> AppConfig:
    console.print("[bold yellow]No configuration found or it's invalid. Let's set it up![/bold yellow]")
    server_url = Prompt.ask("Enter your API server URL (e.g., http://localhost)")
    server_port = Prompt.ask("Enter your API server port (e.g., 8000)", default="8000", show_default=True)
    
    while not server_port.isdigit():
        console.print("[bold red]Port must be a number.[/bold red]")
        server_port = Prompt.ask("Enter your API server port (e.g., 8000)", default="8000", show_default=True)

    config = AppConfig(server_url=server_url, server_port=int(server_port))
    save_config(config)
    return config

app = typer.Typer(
    pretty_exceptions_enable=False,  # Disable pretty exceptions to avoid conflicts with Rich
    help="A CLI for interacting with your AI and Audio API."
)

@app.callback()
def main(
    ctx: typer.Context,
    interactive: Annotated[bool, typer.Option("-i", "--interactive", help="Enable interactive mode for setup and enrollment.")] = False
):
    """
    Manages global options and initializes the application.
    """
    config = load_config()
    if not config:
        config = setup_interactive_config()
        
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["interactive"] = interactive

    if not config.token and interactive:
        console.print("[bold blue]It looks like you don't have a user token. Let's enroll you![/bold blue]")
        username = Prompt.ask("Enter a username to enroll")
        try:
            response = requests.post(f"{config.base_url}/messages/enroll", json={"username": username})
            response.raise_for_status()
            data = response.json()
            if "token" in data:
                config.token = data["token"]
                save_config(config)
                console.print(f"[bold green]Successfully enrolled! Your token is:[/bold green] [yellow]{config.token}[/yellow]")
            else:
                console.print(f"[bold red]Enrollment failed:[/bold red] {data.get('error', 'Unknown error')}")
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Error during enrollment:[/bold red] Could not connect to the API. Please check your server URL and port. ({e})")
            typer.Exit(code=1)
            

@app.command()
def ask(
    ctx: typer.Context,
    prompt: Annotated[str, typer.Argument(help="The prompt to send to the AI.")]
):
    """
    Ask the AI a question.
    """
    config: AppConfig = ctx.obj["config"]
    if not config.token:
        console.print("[bold red]Error: No token found. Please run with -i or enroll a user first.[/bold red]")
        raise typer.Exit(code=1)

    try:
        console.print(f"[bold cyan]Sending prompt:[/bold cyan] {prompt}")
        response = requests.post(f"{config.base_url}/messages/ask", json={"prompt": prompt, "token": config.token})
        response.raise_for_status()
        result = response.json()
        if "message" in result:
            console.print(f"[bold green]AI Response:[/bold green] {result['message']}")
        else:
            console.print(f"[bold red]Error from API:[/bold red] {result.get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error connecting to API:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def enroll(
    ctx: typer.Context,
    username: Annotated[str, typer.Argument(help="The username to enroll.")],
):
    """
    Enroll a new user and get a token.
    """
    config: AppConfig = ctx.obj["config"]
    try:
        console.print(f"[bold cyan]Attempting to enroll user:[/bold cyan] {username}")
        response = requests.post(f"{config.base_url}/messages/enroll", json={"username": username})
        response.raise_for_status()
        data = response.json()
        if "token" in data:
            config.token = data["token"]
            save_config(config)
            console.print(f"[bold green]Successfully enrolled! Your token is:[/bold green] [yellow]{config.token}[/yellow]")
        else:
            console.print(f"[bold red]Enrollment failed:[/bold red] {data.get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error connecting to API:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def remove(
    ctx: typer.Context,
    token_to_remove: Annotated[str, typer.Argument(help="The token of the user to remove.")] = None
):
    """
    Remove a user from the database. If no token is provided, uses the current configured token.
    """
    config: AppConfig = ctx.obj["config"]
    token = token_to_remove if token_to_remove else config.token

    if not token:
        console.print("[bold red]Error: No token specified to remove, and no token is configured.[/bold red]")
        raise typer.Exit(code=1)
        
    if not Confirm.ask(f"Are you sure you want to remove token [yellow]{token}[/yellow]? This cannot be undone.", default=False):
        console.print("[bold yellow]Removal cancelled.[/bold yellow]")
        raise typer.Exit()

    try:
        console.print(f"[bold cyan]Attempting to remove token:[/bold cyan] {token}")
        response = requests.post(f"{config.base_url}/remove", json={"token": token})
        response.raise_for_status()
        result = response.json()
        if "message" in result:
            console.print(f"[bold green]Success:[/bold green] {result['message']}")
            if token == config.token:
                config.token = None # Clear the token if it was the current one
                save_config(config)
                console.print("[bold yellow]Your configured token has been removed. Please enroll a new user.[/bold yellow]")
        else:
            console.print(f"[bold red]Error from API:[/bold red] {result.get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error connecting to API:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def transcribe(
    ctx: typer.Context,
    audio_file: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, help="Path to the audio file (mp3 or wav).")]
):
    """
    Transcribe an audio file.
    """
    config: AppConfig = ctx.obj["config"]
    try:
        console.print(f"[bold cyan]Uploading and transcribing:[/bold cyan] {audio_file}")
        with open(audio_file, "rb") as f:
            files = {"file": (audio_file.name, f, "audio/mpeg" if audio_file.suffix == ".mp3" else "audio/wav")}
            response = requests.post(f"{config.base_url}/audio/transcribe", files=files)
        response.raise_for_status()
        result = response.json()
        if "transcript" in result:
            console.print(f"[bold green]Transcript:[/bold green] {result['transcript']}")
        else:
            console.print(f"[bold red]Error from API:[/bold red] {result.get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error connecting to API:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command("audio-enroll")
def audio_enroll(
    ctx: typer.Context,
    audio_file: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, help="Path to the audio file (mp3 or wav).")],
    token: Annotated[str, typer.Option(help="The token of the user to enroll for voice identification. Uses configured token if not provided.")] = None
):
    """
    Enroll a user's voice for identification.
    """
    config: AppConfig = ctx.obj["config"]
    target_token = token if token else config.token
    
    if not target_token:
        console.print("[bold red]Error: No token provided or configured for voice enrollment.[/bold red]")
        raise typer.Exit(code=1)

    try:
        console.print(f"[bold cyan]Uploading and enrolling voice for token:[/bold cyan] {target_token}")
        with open(audio_file, "rb") as f:
            files = {"file": (audio_file.name, f, "audio/mpeg" if audio_file.suffix == ".mp3" else "audio/wav")}
            data = {"token": target_token}
            response = requests.post(f"{config.base_url}/audio/enroll", files=files, data=data)
        response.raise_for_status()
        result = response.json()
        if "message" in result:
            console.print(f"[bold green]Success:[/bold green] {result['message']}")
        else:
            console.print(f"[bold red]Error from API:[/bold red] {result.get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error connecting to API:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command("transcribe-identify")
def transcribe_identify(
    ctx: typer.Context,
    audio_file: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, help="Path to the audio file (mp3 or wav).")]
):
    """
    Transcribe an audio file and identify the speaker.
    """
    config: AppConfig = ctx.obj["config"]
    try:
        console.print(f"[bold cyan]Uploading, transcribing, and identifying:[/bold cyan] {audio_file}")
        with open(audio_file, "rb") as f:
            files = {"file": (audio_file.name, f, "audio/mpeg" if audio_file.suffix == ".mp3" else "audio/wav")}
            response = requests.post(f"{config.base_url}/audio/transcribe_identify", files=files)
        response.raise_for_status()
        result = response.json()
        if "transcript" in result and "identified_speaker" in result:
            console.print(f"[bold green]Transcript:[/bold green] {result['transcript']}")
            console.print(f"[bold green]Identified Speaker:[/bold green] {result['identified_speaker']}")
        else:
            console.print(f"[bold red]Error from API or unexpected response:[/bold red] {result.get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error connecting to API:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def tts(
    ctx: typer.Context,
    text: Annotated[str, typer.Argument(help="The text to convert to speech.")],
    output_path: Annotated[Path, typer.Option("-o", "--output", help="Path to save the generated MP3 file.")] = Path("output.mp3")
):
    """
    Convert text to speech and save as an MP3 file.
    """
    config: AppConfig = ctx.obj["config"]
    try:
        console.print(f"[bold cyan]Converting text to speech:[/bold cyan] '{text}'")
        response = requests.post(f"{config.base_url}/audio/tts", params={"text": text}, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        console.print(f"[bold green]TTS audio saved to:[/bold green] [yellow]{output_path.resolve()}[/yellow]")
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error connecting to API:[/bold red] {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()