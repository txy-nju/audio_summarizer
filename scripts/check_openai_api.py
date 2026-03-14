
import os
from dotenv import load_dotenv
from pathlib import Path
import openai

def check_api():
    """
    A simple script to check if the OpenAI API is reachable.
    """
    print("--- OpenAI API Connection Check ---")

    # 1. Load environment variables from .env file
    print("Loading environment variables from .env file...")
    
    # 强制覆盖系统环境变量，确保使用的是项目根目录下的 .env 文件内容
    # 由于该文件现在位于 scripts/ 目录下，向上退一级即可到达项目根目录
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / '.env'
    
    if env_path.exists():
        print(f"Found .env file at {env_path.absolute()}")
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        print(f"[WARNING] .env file not found at {env_path.absolute()}. Falling back to system environment variables.")
        load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") # Optional, for proxy/mirror
    
    if not api_key:
        print("\n[ERROR] OPENAI_API_KEY not found in .env file or environment variables.")
        print("Please make sure your .env file is in the project root and contains the key.")
        return

    print(f"API Key loaded (last 4 chars): ...{api_key[-4:]}")
    # print(f"Full API Key (Debug): {api_key}") # Uncomment this line only for debugging, never commit!

    if base_url:
        print(f"Using custom Base URL: {base_url}")
    else:
        print("Using default OpenAI Base URL.")

    # 2. Initialize OpenAI client
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize OpenAI client: {e}")
        return

    # 3. Make a lightweight API call
    print("\nAttempting to connect to the API by listing models...")
    try:
        response = client.models.list()
        # If successful, print some model names
        print("\n[SUCCESS] API connection successful!")
        print("Available models (sample):")
        for model in list(response.data)[:5]:
            print(f"- {model.id}")
            
    except openai.APIConnectionError as e:
        print("\n[ERROR] API Connection Error: Failed to connect to the server.")
        print("This is likely a network issue. Please check:")
        print("1. Your internet connection.")
        print("2. If you are in a restricted region, ensure your VPN or proxy is active.")
        print("3. If using a proxy, ensure your HTTP_PROXY/HTTPS_PROXY environment variables are set correctly.")
        print(f"Underlying error: {e.__cause__}")
        
    except openai.AuthenticationError as e:
        print("\n[ERROR] Authentication Error: Your API key is incorrect or invalid.")
        print("Please check your OPENAI_API_KEY in the .env file.")
        
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {type(e).__name__}")
        print(f"Details: {e}")

if __name__ == "__main__":
    check_api()
