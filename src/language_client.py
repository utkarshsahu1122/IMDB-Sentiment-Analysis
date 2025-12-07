import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

load_dotenv()


class LanguageService:
    """
    Thin wrapper around Azure AI Language (Text Analytics) for sentiment analysis.
    """

    def __init__(self) -> None:
        endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
        key = os.getenv("AZURE_LANGUAGE_KEY")

        if not endpoint or not key:
            raise ValueError("AZURE_LANGUAGE_ENDPOINT or AZURE_LANGUAGE_KEY not set in .env")

        self.client = TextAnalyticsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

    @staticmethod
    def _truncate(text: str, max_chars: int = 5000) -> str:
        """
        Azure Language has a max document size; keep it safe.
        """
        text = text or ""
        if len(text) > max_chars:
            return text[:max_chars]
        return text

    def analyze_sentiment_batch(self, texts: List[str], language: str = "en") -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.

        Returns a list of dicts with:
        - sentiment: 'positive' | 'neutral' | 'negative' | 'mixed'
        - confidence_scores: dict with positive/neutral/negative
        """
        if not texts:
            return []

        docs = [self._truncate(t) for t in texts]

        response = self.client.analyze_sentiment(
            docs,
            language=language,
            show_opinion_mining=False,
        )

        results: List[Dict[str, Any]] = []
        for doc in response:
            if doc.is_error:
                results.append(
                    {
                        "error": True,
                        "code": doc.error.code,
                        "message": doc.error.message,
                    }
                )
            else:
                results.append(
                    {
                        "error": False,
                        "sentiment": doc.sentiment,
                        "confidence_scores": {
                            "positive": doc.confidence_scores.positive,
                            "neutral": doc.confidence_scores.neutral,
                            "negative": doc.confidence_scores.negative,
                        },
                    }
                )
        return results
