import json
import datetime
from google.genai import types

def load_config():
    """Loads config.json from disk."""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config.json: {e}")
        return None

def prepare_system_prompt(config: dict) -> str:
    """Injects dynamic variables (like today's date) into the system prompt."""
    prompt = config.get("system_prompt", "You are a helpful assistant.")
    today_date = datetime.date.today()
    prompt = prompt.replace("{TODAY_DATE}", today_date.strftime("%Y-%m-%d"))
    prompt = prompt.replace("{TODAY_DAY_OF_WEEK}", today_date.strftime("%A"))
    return prompt

def parse_gemini_tools(config: dict) -> list:
    """Parses JSON tool schemas into Gemini Tool objects."""
    gemini_tools = []
    for tool_dict in config.get("tools", []):
        fn = tool_dict.get("function", {})
        try:
            gemini_tools.append(types.Tool(function_declarations=[types.FunctionDeclaration(**fn)]))
        except Exception as e:
            print(f"Error parsing tool schema: {e}")
    return gemini_tools
