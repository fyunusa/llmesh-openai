<?php

declare(strict_types=1);

namespace LLMesh\OpenAI;

/**
 * Enum of OpenAI models with their capabilities.
 */
enum OpenAIModelEnum: string
{
    case GPT4O = 'gpt-4o';
    case GPT4_TURBO = 'gpt-4-turbo';
    case GPT35_TURBO = 'gpt-3.5-turbo';
    case O1 = 'o1';
    case O1_MINI = 'o1-mini';
    case EMBEDDING_3_SMALL = 'text-embedding-3-small';
    case EMBEDDING_3_LARGE = 'text-embedding-3-large';

    /**
     * Check if this model supports tool calling.
     */
    public function supportsTools(): bool
    {
        return match ($this) {
            self::GPT4O, self::GPT4_TURBO, self::GPT35_TURBO => true,
            self::O1, self::O1_MINI => false, // o1 models don't support tools yet
            self::EMBEDDING_3_SMALL, self::EMBEDDING_3_LARGE => false,
        };
    }

    /**
     * Check if this model supports streaming.
     */
    public function supportsStreaming(): bool
    {
        return match ($this) {
            self::GPT4O, self::GPT4_TURBO, self::GPT35_TURBO => true,
            self::O1, self::O1_MINI => false, // o1 models don't support streaming
            self::EMBEDDING_3_SMALL, self::EMBEDDING_3_LARGE => false,
        };
    }

    /**
     * Check if this is an embedding model.
     */
    public function isEmbedding(): bool
    {
        return match ($this) {
            self::EMBEDDING_3_SMALL, self::EMBEDDING_3_LARGE => true,
            default => false,
        };
    }
}
