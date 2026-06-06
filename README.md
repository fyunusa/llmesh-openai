# LLMesh OpenAI Provider

[![Latest Stable Version](https://poser.pugx.org/llmesh/openai/v)](https://packagist.org/packages/llmesh/openai)
[![PHP Version](https://img.shields.io/badge/php-%5E8.1-blue.svg)](https://packagist.org/packages/llmesh/openai)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An OpenAI provider adapter for the [LLMesh Core](https://github.com/fyunusa/llmesh) framework, allowing you to use OpenAI models seamlessly with a unified API.

---

## Installation

Install via Composer:

```bash
composer require llmesh/openai
```

---

## Quick Start

```php
use LLMesh\OpenAI\OpenAIProvider;
use LLMesh\Core\LLMesh;
use LLMesh\Core\Generators\GenerateTextOptions;

$provider = new OpenAIProvider(getenv('OPENAI_API_KEY'));

// Simple Text Generation
$response = LLMesh::generateText(
    $provider,
    GenerateTextOptions::make()->withPrompt('Say hello!')
);

echo $response->getText();
```

---

## Supported Capabilities

- **Chat Completions**: standard, streaming, and tool calls using models like `gpt-4o`, `gpt-4-turbo`, `o1`, and `gpt-3.5-turbo`.
- **Structured Outputs**: validated object generation via native tool mode or JSON mode.
- **Embeddings**: single embedding and batch embedding support (`text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`).
