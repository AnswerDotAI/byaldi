from typing import Any, List

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

from byaldi.objects import Result


class ByaldiLangChainRetriever(BaseRetriever):
    model: Any
    kwargs: dict = {}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa
    ) -> List[Result]:
        """Get documents relevant to a query."""
        docs = self.model.search(query, **self.kwargs)
        return docs