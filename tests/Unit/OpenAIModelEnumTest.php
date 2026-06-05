<?php

declare(strict_types=1);

namespace LLMesh\OpenAI\Tests\Unit;

use LLMesh\OpenAI\OpenAIModelEnum;
use PHPUnit\Framework\TestCase;

final class OpenAIModelEnumTest extends TestCase
{
    public function testGpt4oSupportsToolsAndStreaming(): void
    {
        $model = OpenAIModelEnum::GPT4O;

        $this->assertTrue($model->supportsTools());
        $this->assertTrue($model->supportsStreaming());
        $this->assertFalse($model->isEmbedding());
    }

    public function testGpt4TurboSupportsToolsAndStreaming(): void
    {
        $model = OpenAIModelEnum::GPT4_TURBO;

        $this->assertTrue($model->supportsTools());
        $this->assertTrue($model->supportsStreaming());
    }

    public function testGpt35TurboSupportsToolsAndStreaming(): void
    {
        $model = OpenAIModelEnum::GPT35_TURBO;

        $this->assertTrue($model->supportsTools());
        $this->assertTrue($model->supportsStreaming());
    }

    public function testO1DoesNotSupportToolsOrStreaming(): void
    {
        $model = OpenAIModelEnum::O1;

        $this->assertFalse($model->supportsTools());
        $this->assertFalse($model->supportsStreaming());
    }

    public function testO1MiniDoesNotSupportToolsOrStreaming(): void
    {
        $model = OpenAIModelEnum::O1_MINI;

        $this->assertFalse($model->supportsTools());
        $this->assertFalse($model->supportsStreaming());
    }

    public function testEmbedding3SmallIsEmbeddingModel(): void
    {
        $model = OpenAIModelEnum::EMBEDDING_3_SMALL;

        $this->assertTrue($model->isEmbedding());
        $this->assertFalse($model->supportsTools());
        $this->assertFalse($model->supportsStreaming());
    }

    public function testEmbedding3LargeIsEmbeddingModel(): void
    {
        $model = OpenAIModelEnum::EMBEDDING_3_LARGE;

        $this->assertTrue($model->isEmbedding());
        $this->assertFalse($model->supportsTools());
        $this->assertFalse($model->supportsStreaming());
    }

    public function testEnumValuesAreCorrect(): void
    {
        $this->assertSame('gpt-4o', OpenAIModelEnum::GPT4O->value);
        $this->assertSame('gpt-4-turbo', OpenAIModelEnum::GPT4_TURBO->value);
        $this->assertSame('gpt-3.5-turbo', OpenAIModelEnum::GPT35_TURBO->value);
        $this->assertSame('o1', OpenAIModelEnum::O1->value);
        $this->assertSame('o1-mini', OpenAIModelEnum::O1_MINI->value);
        $this->assertSame('text-embedding-3-small', OpenAIModelEnum::EMBEDDING_3_SMALL->value);
        $this->assertSame('text-embedding-3-large', OpenAIModelEnum::EMBEDDING_3_LARGE->value);
    }
}
