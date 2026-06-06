<?php

declare(strict_types=1);

namespace LLMesh\OpenAI\Tests\Unit;

use LLMesh\OpenAI\OpenAIProvider;
use LLMesh\OpenAI\OpenAIModelEnum;
use LLMesh\Core\Data\Message;
use LLMesh\Core\Data\MessageRole;
use LLMesh\Core\Exceptions\RateLimitException;
use LLMesh\Core\Exceptions\ProviderException;
use LLMesh\Core\Exceptions\TokenLimitException;
use LLMesh\Core\Exceptions\HttpException;
use LLMesh\Core\Http\HttpClient;
use PHPUnit\Framework\TestCase;

final class OpenAIProviderTest extends TestCase
{
    private OpenAIProvider $provider;
    private HttpClient $mockHttpClient;

    protected function setUp(): void
    {
        $this->mockHttpClient = $this->createMock(HttpClient::class);
        $this->provider = new OpenAIProvider(
            apiKey: 'sk-test-key-12345',
            model: 'gpt-4o',
            httpClient: $this->mockHttpClient,
        );
    }

    public function testCanMakeChatRequest(): void
    {
        $messages = [
            Message::user('Hello'),
        ];

        $this->mockHttpClient
            ->expects($this->once())
            ->method('post')
            ->with(
                '/chat/completions',
                $this->callback(function ($payload) {
                    return $payload['model'] === 'gpt-4o'
                        && $payload['messages'][0]['role'] === 'user'
                        && $payload['messages'][0]['content'] === 'Hello';
                }),
                $this->isType('array'),
            )
            ->willReturn([
                'choices' => [
                    [
                        'message' => ['content' => 'Hi there!'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'usage' => [
                    'prompt_tokens' => 10,
                    'completion_tokens' => 5,
                ],
            ]);

        $response = $this->provider->chat($messages);

        $this->assertSame('Hi there!', $response->getText());
        $this->assertSame('stop', $response->getFinishReason());
        $this->assertSame(10, $response->getUsage()->getInputTokens());
        $this->assertSame(5, $response->getUsage()->getOutputTokens());
    }

    public function testPassesOptionsToRequest(): void
    {
        $messages = [Message::user('Test')];

        $this->mockHttpClient
            ->expects($this->once())
            ->method('post')
            ->with(
                '/chat/completions',
                $this->callback(function ($payload) {
                    return isset($payload['temperature'], $payload['max_tokens'])
                        && $payload['temperature'] === 0.7
                        && $payload['max_tokens'] === 500;
                }),
                $this->isType('array'),
            )
            ->willReturn([
                'choices' => [['message' => ['content' => 'Test'], 'finish_reason' => 'stop']],
                'usage' => ['prompt_tokens' => 0, 'completion_tokens' => 0],
            ]);

        $this->provider->chat($messages, [
            'temperature' => 0.7,
            'max_tokens' => 500,
        ]);
    }

    public function testHandlesToolCallsInResponse(): void
    {
        $messages = [Message::user('Get weather')];

        $this->mockHttpClient
            ->expects($this->once())
            ->method('post')
            ->willReturn([
                'choices' => [
                    [
                        'message' => [
                            'content' => null,
                            'tool_calls' => [
                                [
                                    'id' => 'call_123',
                                    'function' => [
                                        'name' => 'get_weather',
                                        'arguments' => '{"city": "NYC"}',
                                    ],
                                ],
                            ],
                        ],
                        'finish_reason' => 'tool_calls',
                    ],
                ],
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ]);

        $response = $this->provider->chat($messages);

        $this->assertSame('tool_calls', $response->getFinishReason());
    }

    public function testThrowsRateLimitExceptionOn429(): void
    {
        $this->mockHttpClient
            ->expects($this->once())
            ->method('post')
            ->willThrowException(new HttpException('Too many requests', 429, '{"error": {"retry_after_ms": 5000}}'));

        $this->expectException(RateLimitException::class);
        $this->expectExceptionMessage('OpenAI rate limit exceeded');

        $this->provider->chat([Message::user('Test')]);
    }

    public function testThrowsTokenLimitExceptionOnContextLengthError(): void
    {
        $this->mockHttpClient
            ->expects($this->once())
            ->method('post')
            ->willThrowException(new HttpException(
                'Bad request',
                400,
                json_encode([
                    'error' => [
                        'code' => 'context_length_exceeded',
                        'message' => 'This model\'s maximum context length is 8192 tokens',
                    ],
                ]),
            ));

        $this->expectException(TokenLimitException::class);

        $this->provider->chat([Message::user('Test')]);
    }

    public function testThrowsProviderExceptionOn400(): void
    {
        $this->mockHttpClient
            ->expects($this->once())
            ->method('post')
            ->willThrowException(new HttpException(
                'Bad request',
                400,
                json_encode(['error' => ['message' => 'Invalid request']]),
            ));

        $this->expectException(ProviderException::class);

        $this->provider->chat([Message::user('Test')]);
    }

    public function testSupportsCapabilities(): void
    {
        $this->assertTrue($this->provider->supports('streaming'));
        $this->assertTrue($this->provider->supports('tools'));
        $this->assertTrue($this->provider->supports('embeddings'));
        $this->assertFalse($this->provider->supports('unknown'));
    }

    public function testCanMakeEmbedRequest(): void
    {
        $this->mockHttpClient
            ->expects($this->once())
            ->method('post')
            ->with(
                '/embeddings',
                $this->callback(function ($payload) {
                    return $payload['model'] === 'text-embedding-3-small'
                        && $payload['input'] === 'Hello world';
                }),
                $this->isType('array'),
            )
            ->willReturn([
                'data' => [
                    [
                        'embedding' => [0.1, 0.2, 0.3],
                        'index' => 0,
                    ],
                ],
                'model' => 'text-embedding-3-small',
                'usage' => ['prompt_tokens' => 5],
            ]);

        $response = $this->provider->embed('Hello world');

        $this->assertSame([0.1, 0.2, 0.3], $response->getEmbedding());
        $this->assertSame(3, $response->getDimensions());
    }

    public function testCanMakeBatchEmbedRequest(): void
    {
        $inputs = ['Hello', 'World'];

        $this->mockHttpClient
            ->expects($this->once())
            ->method('post')
            ->with(
                '/embeddings',
                $this->callback(function ($payload) use ($inputs) {
                    return $payload['input'] === $inputs;
                }),
                $this->isType('array'),
            )
            ->willReturn([
                'data' => [
                    ['embedding' => [0.1, 0.2], 'index' => 0],
                    ['embedding' => [0.3, 0.4], 'index' => 1],
                ],
                'model' => 'text-embedding-3-small',
                'usage' => ['prompt_tokens' => 10],
            ]);

        $response = $this->provider->embed($inputs);

        $this->assertSame([0.1, 0.2], $response->getEmbedding());
    }

    public function testAuthorizationHeaderIncludesApiKey(): void
    {
        $this->mockHttpClient
            ->expects($this->once())
            ->method('post')
            ->with(
                $this->isType('string'),
                $this->isType('array'),
                $this->callback(function ($headers) {
                    return isset($headers['Authorization'])
                        && strpos($headers['Authorization'], 'Bearer') === 0;
                }),
            )
            ->willReturn([
                'choices' => [['message' => ['content' => 'Test'], 'finish_reason' => 'stop']],
                'usage' => ['prompt_tokens' => 0, 'completion_tokens' => 0],
            ]);

        $this->provider->chat([Message::user('Test')]);
    }

    // -------------------------------------------------------------------------
    // Batch embedding — single API call for array input
    // -------------------------------------------------------------------------

    public function testBatchEmbedMakesASingleApiCall(): void
    {
        $inputs = ['Hello', 'World', 'Foo'];

        // Only ONE HTTP POST must be made for the whole batch
        $this->mockHttpClient
            ->expects($this->once())
            ->method('post')
            ->with(
                '/embeddings',
                $this->callback(fn ($payload) => $payload['input'] === $inputs),
                $this->isType('array'),
            )
            ->willReturn([
                'data'  => [
                    ['embedding' => [0.1, 0.2], 'index' => 0],
                    ['embedding' => [0.3, 0.4], 'index' => 1],
                    ['embedding' => [0.5, 0.6], 'index' => 2],
                ],
                'model' => 'text-embedding-3-small',
                'usage' => ['prompt_tokens' => 15],
            ]);

        $response = $this->provider->embed($inputs);

        // embed() returns the first embedding (index 0) per ProviderInterface contract
        $this->assertSame([0.1, 0.2], $response->getEmbedding());
    }

    // -------------------------------------------------------------------------
    // parseBatchEmbeddingResponse — correct index mapping
    // -------------------------------------------------------------------------

    public function testParseBatchEmbeddingResponseReturnsCorrectlyIndexedResults(): void
    {
        $raw = [
            'data'  => [
                ['embedding' => [0.1, 0.2], 'index' => 0],
                ['embedding' => [0.3, 0.4], 'index' => 1],
            ],
            'model' => 'text-embedding-3-small',
            'usage' => ['prompt_tokens' => 10],
        ];

        $results = $this->provider->parseBatchEmbeddingResponse($raw, ['Hello', 'World']);

        $this->assertCount(2, $results);
        $this->assertSame([0.1, 0.2], $results[0]->getEmbedding());
        $this->assertSame([0.3, 0.4], $results[1]->getEmbedding());
    }

    public function testParseBatchEmbeddingResponseSortsOutOfOrderIndices(): void
    {
        // OpenAI may return results in any order — we must sort by index field
        $raw = [
            'data'  => [
                ['embedding' => [0.9, 0.8], 'index' => 1],
                ['embedding' => [0.1, 0.2], 'index' => 0],
            ],
            'model' => 'text-embedding-3-small',
            'usage' => ['prompt_tokens' => 10],
        ];

        $results = $this->provider->parseBatchEmbeddingResponse($raw);

        $this->assertSame([0.1, 0.2], $results[0]->getEmbedding());
        $this->assertSame([0.9, 0.8], $results[1]->getEmbedding());
    }

    // -------------------------------------------------------------------------
    // Custom model for embeddings
    // -------------------------------------------------------------------------

    public function testEmbedUsesCustomModel(): void
    {
        $this->mockHttpClient
            ->expects($this->once())
            ->method('post')
            ->with(
                '/embeddings',
                $this->callback(fn ($p) => $p['model'] === 'text-embedding-3-large'),
                $this->isType('array'),
            )
            ->willReturn([
                'data'  => [['embedding' => [0.5], 'index' => 0]],
                'model' => 'text-embedding-3-large',
                'usage' => ['prompt_tokens' => 3],
            ]);

        $this->provider->embed('hello', ['model' => 'text-embedding-3-large']);
    }

    public function testCanStreamChat(): void
    {
        $messages = [Message::user('Hello')];

        $streamLines = [
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            'data: {"choices":[{"delta":{"content":" there"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            'data: [DONE]'
        ];

        $generatorHelper = function() use ($streamLines) {
            foreach ($streamLines as $line) {
                yield $line;
            }
        };

        $this->mockHttpClient
            ->expects($this->once())
            ->method('stream')
            ->willReturn($generatorHelper());

        $stream = $this->provider->stream($messages);
        $chunks = iterator_to_array($stream->getChunks());

        $this->assertCount(3, $chunks);
        $this->assertSame('Hi', $chunks[0]->text);
        $this->assertSame(' there', $chunks[1]->text);
        $this->assertNull($chunks[2]->text);
        $this->assertSame('stop', $chunks[2]->finishReason);
    }
}

