"""Production dependency wiring selected from application settings."""

from collections.abc import Callable

from jesseagent.agent.context import AgentContextBudget
from jesseagent.agent.orchestrator import AgentSession
from jesseagent.agent.runs import AgentRun
from jesseagent.application.agent_runs.service import AgentRunService
from jesseagent.application.embedding import EmbeddingProvider
from jesseagent.application.knowledge.chunking import chunk_markdown
from jesseagent.application.knowledge.search import KnowledgeSearchService
from jesseagent.application.knowledge.sync import KnowledgeSyncService
from jesseagent.application.video.contracts import (
    ChatProvider,
    SummaryProvider,
    TranscriptIndexRepository,
    VisionAnalyzer,
    VisionIndexRepository,
)
from jesseagent.application.video.service import VideoService
from jesseagent.core.config import Settings, settings
from jesseagent.domain.transcript_index import TranscriptChunkPolicy
from jesseagent.infrastructure.chroma.knowledge import ChromaKnowledgeIndex
from jesseagent.infrastructure.chroma.transcript import (
    ChromaTranscriptIndexRepository,
)
from jesseagent.infrastructure.chroma.vision import (
    ChromaVisionIndexRepository,
)
from jesseagent.infrastructure.gemini.agent_model import GeminiAgentModel
from jesseagent.infrastructure.gemini.chat import GeminiChatProvider
from jesseagent.infrastructure.gemini.embedding import GeminiEmbeddingProvider
from jesseagent.infrastructure.gemini.summary import GeminiSummaryProvider
from jesseagent.infrastructure.gemini.vision import GeminiVisionAnalyzer
from jesseagent.infrastructure.local.video_cache import (
    CacheFreshnessPolicy,
    LocalCacheManager,
)
from jesseagent.infrastructure.sqlite.agent_runs import (
    SQLiteAgentRunRepository,
)
from jesseagent.infrastructure.sqlite.knowledge import (
    SQLiteKnowledgeCatalog,
)
from jesseagent.infrastructure.youtube.loader import YouTubeLoader
from jesseagent.tools.contracts import ToolResult
from jesseagent.tools.executor import ToolExecutor


def create_video_service(config: Settings = settings) -> VideoService:
    """Build the production service using the configured infrastructure."""
    return VideoService(
        cache=LocalCacheManager(
            data_dir=config.data_dir,
            freshness_policy=CacheFreshnessPolicy(
                summary_model=config.summary_model,
                summary_prompt_version=config.summary_prompt_version,
                summary_language=config.summary_language,
                summary_chapter_window_policy=config.chapter_window_policy.cache_key,
                summary_chapter_block_policy=config.chapter_block_policy.cache_key,
                vision_model=config.vision_model,
                vision_prompt_version=config.vision_prompt_version,
                embedding_model=config.embedding_model,
                embedding_dimension=config.embedding_dimension,
            ),
        ),
        loader=YouTubeLoader(),
        embedding_provider_factory=_embedding_provider_factory(config),
        transcript_index_repository_factory=_repository_factory(config),
        summary_provider_factory=_summary_provider_factory(config),
        vision_analyzer_factory=_vision_analyzer_factory(config),
        vision_index_repository_factory=_vision_repository_factory(config),
        chat_provider_factory=_chat_provider_factory(config),
        summary_model=config.summary_model,
        summary_prompt_version=config.summary_prompt_version,
        summary_language=config.summary_language,
        summary_chapter_window_policy=config.chapter_window_policy.cache_key,
        summary_chapter_block_policy=config.chapter_block_policy.cache_key,
        vision_model=config.vision_model,
        vision_prompt_version=config.vision_prompt_version,
    )


def create_agent_session(
    config: Settings = settings,
    on_tool_result: Callable[[ToolResult], None] | None = None,
) -> AgentSession:
    """Build the human-facing Agent with a fresh tool and chat-session scope."""
    return AgentSession(
        model=GeminiAgentModel(
            api_key=config.gemini_api_key,
            model=config.llm_model,
            prompt_version=config.agent_prompt_version,
        ),
        tools=ToolExecutor(
            create_video_service(config), create_knowledge_search_service(config)
        ),
        max_steps=config.agent_max_steps,
        on_tool_result=on_tool_result,
        repository=SQLiteAgentRunRepository(config.data_dir / "agent_runs.sqlite3"),
        context_budget=AgentContextBudget(
            max_messages=config.agent_context_max_messages,
            max_characters=config.agent_context_max_characters,
        ),
    )


def create_agent_run_service(config: Settings = settings) -> AgentRunService:
    """Build the durable run API used by CLI and future trigger adapters."""
    repository = SQLiteAgentRunRepository(config.data_dir / "agent_runs.sqlite3")

    def sessions(run: AgentRun | None = None) -> AgentSession:
        return AgentSession(
            model=GeminiAgentModel(
                config.gemini_api_key, config.llm_model, config.agent_prompt_version
            ),
            tools=ToolExecutor(
                create_video_service(config), create_knowledge_search_service(config)
            ),
            max_steps=config.agent_max_steps,
            repository=repository,
            existing_run=run,
            context_budget=AgentContextBudget(
                max_messages=config.agent_context_max_messages,
                max_characters=config.agent_context_max_characters,
            ),
        )

    return AgentRunService(repository, sessions)


def create_knowledge_sync_service(config: Settings = settings) -> KnowledgeSyncService:
    """Build the source-neutral incremental indexing service."""
    return KnowledgeSyncService(
        SQLiteKnowledgeCatalog(config.data_dir / "knowledge.sqlite3"),
        ChromaKnowledgeIndex(
            config.data_dir / "knowledge_chromadb",
            config.embedding_model,
            config.embedding_dimension,
        ),
        _embedding_provider_factory(config),
        chunk_markdown,
    )


def create_knowledge_search_service(
    config: Settings = settings,
) -> KnowledgeSearchService:
    return KnowledgeSearchService(
        SQLiteKnowledgeCatalog(config.data_dir / "knowledge.sqlite3"),
        ChromaKnowledgeIndex(
            config.data_dir / "knowledge_chromadb",
            config.embedding_model,
            config.embedding_dimension,
        ),
        _embedding_provider_factory(config)(),
    )


def _embedding_provider_factory(
    config: Settings,
) -> Callable[[], EmbeddingProvider]:
    match config.embedding_provider:
        case "gemini":
            return lambda: GeminiEmbeddingProvider(
                api_key=config.gemini_api_key,
                model=config.embedding_model,
                dimension=config.embedding_dimension,
            )


def _repository_factory(
    config: Settings,
) -> Callable[[str], TranscriptIndexRepository]:
    match config.vector_repository:
        case "chroma":
            return lambda video_id: ChromaTranscriptIndexRepository(
                video_id,
                data_dir=config.data_dir,
                embedding_model=config.embedding_model,
                embedding_dimension=config.embedding_dimension,
                chunk_policy=TranscriptChunkPolicy(
                    max_seconds=config.transcript_chunk_max_seconds,
                    max_characters=config.transcript_chunk_max_characters,
                ),
            )


def _summary_provider_factory(config: Settings) -> Callable[[], SummaryProvider]:
    """Build a factory so summary credentials are only resolved when needed."""
    return lambda: GeminiSummaryProvider(
        api_key=config.gemini_api_key,
        model=config.summary_model,
        prompt_version=config.summary_prompt_version,
        chapter_window_policy=config.chapter_window_policy,
        chapter_block_policy=config.chapter_block_policy,
    )


def _vision_analyzer_factory(config: Settings) -> Callable[[], VisionAnalyzer]:
    """Create the configured public-URL vision analyzer lazily."""
    return lambda: GeminiVisionAnalyzer(
        api_key=config.gemini_api_key,
        model=config.vision_model,
        prompt_version=config.vision_prompt_version,
    )


def _vision_repository_factory(
    config: Settings,
) -> Callable[[str], VisionIndexRepository]:
    """Create a video-scoped visual-scene vector repository."""
    return lambda video_id: ChromaVisionIndexRepository(
        video_id,
        data_dir=config.data_dir,
        embedding_model=config.embedding_model,
        embedding_dimension=config.embedding_dimension,
    )


def _chat_provider_factory(config: Settings) -> Callable[[], ChatProvider]:
    """Build the configured grounded-answer provider lazily."""
    return lambda: GeminiChatProvider(
        api_key=config.gemini_api_key,
        model=config.llm_model,
        prompt_version=config.chat_prompt_version,
    )
