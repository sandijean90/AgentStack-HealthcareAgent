import base64
from pathlib import Path
from typing import Optional
import os
import google.generativeai as genai
from dotenv import load_dotenv


class GeminiPolicyAgent:
    """
    A Gemini-backed policy agent that reads a benefits PDF and answers coverage questions.
    Original logic preserved in spirit: load PDF, add it to the prompt, return concise answers.
    """

    def __init__(
        self,
        pdf_path: Optional[str] = None,
        model: str = "gemini-2.5-flash-lite",
        system_prompt: str = (
            "You are an expert insurance agent designed to assist with coverage queries. "
            "Use the provided documents to answer questions about insurance policies. "
            "If the information is not available in the documents, respond with \"I don't know\"."
        ),
        max_output_tokens: int = 1024,
    ) -> None:
        # Ensure .env values are picked up even when the working directory differs.
        self._load_env()
        api_key = self._get_gemini_api_key()
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(model)
        self.max_output_tokens = max_output_tokens
        self.system_prompt = system_prompt

        # Default to the same PDF used by the original policy agent
        self.pdf_path = Path(pdf_path) if pdf_path else Path(__file__).resolve().parent / "2026AnthemgHIPSBC.pdf"
        self.pdf_bytes = self._load_pdf(self.pdf_path)

    @staticmethod
    def _get_gemini_api_key() -> str:
        """
        Load the Gemini API key from common env var names.

        Checks (in order):
        - GEMINI_API_KEY
        - GEMINI_APIKEY
        """
        for key in ("GEMINI_API_KEY", "GEMINI_APIKEY"):
            value = os.environ.get(key)
            if value:
                return value
        raise RuntimeError("Gemini API key not found. Set GEMINI_API_KEY (or GEMINI_APIKEY).")

    @staticmethod
    def _load_env() -> None:
        """
        Load environment variables from the repo .env file plus the current working directory.

        This handles cases where the process is started outside the project root.
        """
        project_env = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(project_env)
        load_dotenv()

    @staticmethod
    def _load_pdf(path: Path) -> bytes:
        if not path.exists():
            raise FileNotFoundError(f"PDF not found at {path.resolve()}")
        return path.read_bytes()

    def answer_query(self, prompt: str) -> str:
        """
        Send the user prompt and the embedded PDF to Gemini and return the text response.
        """
        parts = [
            {"text": self.system_prompt},
            {
                "inline_data": {
                    "mime_type": "application/pdf",
                    "data": self.pdf_bytes,
                }
            },
            {"text": prompt},
        ]

        response = self.model.generate_content(
            parts,
            generation_config={"max_output_tokens": self.max_output_tokens},
        )

        if not response or not response.text:
            return "I don't know"

        # Maintain escaping behavior similar to the original agent
        return response.text.replace("$", r"\\$")
