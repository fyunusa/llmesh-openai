<?php

declare(strict_types=1);

namespace LLMesh\OpenAI;

use LLMesh\Core\Contracts\EmbeddingResponseInterface;
use LLMesh\Core\Contracts\ProviderInterface;
use LLMesh\Core\Contracts\ResponseInterface;
use LLMesh\Core\Contracts\StreamInterface;
use LLMesh\Core\Data\Message;
use LLMesh\Core\Exceptions\HttpException;
use LLMesh\Core\Exceptions\ProviderException;
use LLMesh\Core\Exceptions\RateLimitException;
use LLMesh\Core\Exceptions\TokenLimitException;
use LLMesh\Core\Generators\TextResponse;
use LLMesh\Core\Generators\Usage;
use LLMesh\Core\Http\HttpClient;
use LLMesh\Core\Http\HttpClientFactory;
use LLMesh\Core\Config\ProviderConfig;
use LLMesh\Core\Data\ChunkDelta;
use LLMesh\Core\Data\ToolCall;

/**
 * OpenAI provider for LLMesh.
 *
 * Implements the ProviderInterface for OpenAI's chat completions, embeddings, and streaming APIs.
 */
final class OpenAIProvider implements ProviderInterface
{
    private HttpClient $httpClient;

    /**
     * @param string $apiKey OpenAI API key
     * @param string $model Model to use (default: gpt-4o)
     * @param HttpClient|null $httpClient Custom HTTP client (uses factory if null)
     * @param ProviderConfig|null $config Provider configuration
     */
    public function __construct(
        private readonly string $apiKey,
        private readonly string $model = 'gpt-4o',
        HttpClient|null $httpClient = null,
        private readonly ProviderConfig|null $config = null,
    ) {
        $this->httpClient = $httpClient ?? HttpClientFactory::make();
        $this->httpClient->setBaseUrl('https://api.openai.com/v1');
    }

    public function chat(array $messages, array $options = []): ResponseInterface
    {
        try {
            $payload = $this->buildChatPayload($messages, $options);
            $response = $this->httpClient->post(
                '/chat/completions',
                $payload,
                $this->getHeaders(),
            );

            return TextResponse::fromProviderResponse($response, function ($raw) {
                return $this->parseChatResponse($raw);
            });
        } catch (HttpException $e) {
            $this->handleHttpException($e);
        }
    }

    public function stream(array $messages, array $options = []): StreamInterface
    {
        $payload = $this->buildChatPayload($messages, array_merge($options, ['stream' => true]));

        $generator = function () use ($payload) {
            $lines = $this->httpClient->stream(
                '/chat/completions',
                $payload,
                $this->getHeaders(),
            );

            foreach ($lines as $line) {
                if (strpos($line, 'data: ') === 0) {
                    $json = substr($line, 6);

                    if ($json === '[DONE]') {
                        break;
                    }

                    $data = json_decode($json, true);
                    if ($data === null) {
                        continue;
                    }

                    yield $this->parseStreamChunk($data);
                }
            }
        };

        // Return a StreamResponse - we'll create this in core later
        // For now, wrap the generator
        return new \LLMesh\Core\Generators\StreamResponse($generator());
    }

    public function embed(string|array $input, array $options = []): EmbeddingResponseInterface
    {
        $model = $options['model'] ?? 'text-embedding-3-small';
        $payload = [
            'model' => $model,
            'input' => $input,
            'encoding_format' => $options['encoding_format'] ?? 'float',
        ];

        $response = $this->httpClient->post(
            '/embeddings',
            $payload,
            $this->getHeaders(),
        );

        return $this->parseEmbeddingResponse($response);
    }

    public function supports(string $capability): bool
    {
        return match ($capability) {
            'streaming' => true,
            'tools' => true,
            'embeddings' => true,
            default => false,
        };
    }

    /**
     * Build payload for chat completion request.
     *
     * @param array $messages
     * @param array $options
     * @return array
     */
    private function buildChatPayload(array $messages, array $options = []): array
    {
        $payload = [
            'model' => $this->model,
            'messages' => $this->formatMessages($messages),
        ];

        // Add optional parameters if provided
        if (isset($options['temperature'])) {
            $payload['temperature'] = $options['temperature'];
        }

        if (isset($options['max_tokens'])) {
            $payload['max_tokens'] = $options['max_tokens'];
        }

        if (isset($options['stop'])) {
            $payload['stop'] = $options['stop'];
        }

        if (isset($options['tools']) && !empty($options['tools'])) {
            $payload['tools'] = $this->formatTools($options['tools']);
        }

        if (isset($options['stream'])) {
            $payload['stream'] = $options['stream'];
        }

        return $payload;
    }

    /**
     * Format Message DTOs to OpenAI format.
     *
     * @param array $messages
     * @return array
     */
    private function formatMessages(array $messages): array
    {
        return array_map(function ($message) {
            if ($message instanceof Message) {
                $formatted = [
                    'role' => $message->role->value,
                    'content' => $message->content,
                ];

                if ($message->toolCallId) {
                    $formatted['tool_call_id'] = $message->toolCallId;
                }

                return $formatted;
            }

            // Handle array messages
            return $message;
        }, $messages);
    }

    /**
     * Format tools for OpenAI API.
     *
     * @param array $tools
     * @return array
     */
    private function formatTools(array $tools): array
    {
        return array_map(function ($tool) {
            if (method_exists($tool, 'toArray')) {
                return [
                    'type' => 'function',
                    'function' => $tool->toArray(),
                ];
            }

            return [
                'type' => 'function',
                'function' => $tool,
            ];
        }, $tools);
    }

    /**
     * Parse chat completion response.
     *
     * @param array $response
     * @return array{text: string, usage: array, finishReason: string}
     */
    private function parseChatResponse(array $response): array
    {
        $choice = $response['choices'][0] ?? [];
        $message = $choice['message'] ?? [];
        $finishReason = $choice['finish_reason'] ?? 'stop';

        $text = $message['content'] ?? '';

        // Handle tool calls
        if ($finishReason === 'tool_calls' && isset($message['tool_calls'])) {
            $text = json_encode($message['tool_calls']);
        }

        return [
            'text' => $text,
            'usage' => [
                'input_tokens' => $response['usage']['prompt_tokens'] ?? 0,
                'output_tokens' => $response['usage']['completion_tokens'] ?? 0,
            ],
            'finishReason' => $finishReason,
        ];
    }

    /**
     * Parse streaming chunk from OpenAI.
     *
     * @param array $data
     * @return ChunkDelta
     */
    private function parseStreamChunk(array $data): ChunkDelta
    {
        $choice = $data['choices'][0] ?? [];
        $delta = $choice['delta'] ?? [];

        $text = $delta['content'] ?? null;
        $toolCall = null;
        $finishReason = $choice['finish_reason'] ?? null;

        if (isset($delta['tool_calls'])) {
            $toolCall = new ToolCall(
                id: $delta['tool_calls'][0]['id'] ?? '',
                name: $delta['tool_calls'][0]['function']['name'] ?? '',
                arguments: $delta['tool_calls'][0]['function']['arguments'] ?? [],
            );
        }

        return new ChunkDelta(
            text: $text,
            toolCall: $toolCall,
            finishReason: $finishReason,
        );
    }

    /**
     * Parse embedding response from OpenAI.
     *
     * @param array $response
     * @return EmbeddingResponseInterface
     */
    private function parseEmbeddingResponse(array $response): EmbeddingResponseInterface
    {
        // Handle both single and batch embeddings
        if (isset($response['data'][0])) {
            $embedding = $response['data'][0]['embedding'];
        } else {
            // Single embedding case
            $embedding = $response['embedding'] ?? [];
        }

        return new \LLMesh\Core\Embeddings\EmbeddingResponse(
            embedding: $embedding,
            dimensions: count($embedding),
            usage: new Usage(
                inputTokens: $response['usage']['prompt_tokens'] ?? 0,
                outputTokens: 0,
            ),
            model: $response['model'] ?? $this->model,
        );
    }

    /**
     * Get HTTP headers for OpenAI API.
     *
     * @return array
     */
    private function getHeaders(): array
    {
        return [
            'Authorization' => 'Bearer ' . $this->apiKey,
            'Content-Type' => 'application/json',
        ];
    }

    /**
     * Handle HTTP exceptions and convert to provider-specific exceptions.
     *
     * @param HttpException $e
     * @throws RateLimitException|TokenLimitException|ProviderException
     * @return never
     */
    private function handleHttpException(HttpException $e): never
    {
        $status = $e->statusCode();
        $body = $e->responseBody();

        if ($status === 429) {
            // Rate limited
            $retryAfter = null;
            // Try to get Retry-After from response body
            $data = json_decode($body, true);
            if (isset($data['error']['retry_after_ms'])) {
                $retryAfter = (int)ceil($data['error']['retry_after_ms'] / 1000);
            }

            throw new RateLimitException(
                'OpenAI rate limit exceeded',
                'openai',
                retryAfter: $retryAfter,
            );
        }

        if ($status === 400) {
            // Check for context length error
            $data = json_decode($body, true);
            if (isset($data['error']['code']) && $data['error']['code'] === 'context_length_exceeded') {
                throw new TokenLimitException(
                    $data['error']['message'] ?? 'Context length exceeded',
                    'openai',
                    limit: 128000, // Default for gpt-4o
                    used: 0,
                );
            }

            throw new ProviderException(
                $data['error']['message'] ?? 'Bad Request',
                'openai',
            );
        }

        if ($status >= 500) {
            throw new ProviderException(
                'OpenAI server error: ' . $body,
                'openai',
            );
        }

        throw new ProviderException(
            'OpenAI API error: ' . $body,
            'openai',
        );
    }
}
