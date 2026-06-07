<?php

/**
 * LLMesh Demo 06: Observability Middleware (Logging & Cost Tracking)
 *
 * This script demonstrates how to wrap your OpenAI provider using the LLMesh Middleware stack.
 * We attach logging (using a custom PSR-3 console logger) and cost tracking (to measure
 * exact token usage and USD costs) to both standard and streaming requests.
 */

// Load the bootstrap file to handle autoloading and API key setup.
$apiKey = require __DIR__ . '/bootstrap.php';

use LLMesh\Core\LLMesh;
use LLMesh\Core\Generators\GenerateTextOptions;
use LLMesh\Core\Observability\MiddlewareStack;
use LLMesh\Core\Observability\LoggingMiddleware;
use LLMesh\Core\Observability\CostTrackingMiddleware;
use LLMesh\Core\Observability\UsageTracker;
use LLMesh\Core\Observability\CostCalculator;
use LLMesh\OpenAI\OpenAIProvider;
use Psr\Log\AbstractLogger;

echo "=== LLMesh OpenAI Observability Middleware Demo ===\n\n";

// 1. Configure Pricing
// CostCalculator allows mapping specific models to pricing rates per 1M tokens.
// OpenAI pricing for GPT-4o (e.g. $2.50 per 1M input tokens, $10.00 per 1M output tokens).
CostCalculator::setPricing('gpt-4o', 2.50, 10.00);
CostCalculator::setPricing('gpt-4o-2024-05-13', 2.50, 10.00);
CostCalculator::setPricing('gpt-4o-2024-08-06', 2.50, 10.00);

// 2. Define a Console Logger (PSR-3 Compatible)
// Any PSR-3 logger (like Monolog) works out of the box with LoggingMiddleware.
// We write a simple console-based logger here for demonstration purposes.
class ConsoleLogger extends AbstractLogger
{
    public function log($level, string|\Stringable $message, array $context = []): void
    {
        $time = date('Y-m-d H:i:s');
        $levelUpper = strtoupper((string)$level);
        echo "[{$time}] [{$levelUpper}] {$message}\n";
        if (!empty($context)) {
            // Context contains full payload details (prompt, options, tokens)
            echo "Context info:\n" . json_encode($context, JSON_PRETTY_PRINT) . "\n";
        }
        echo "------------------------------------------------\n";
    }
}

try {
    $rawProvider = new OpenAIProvider($apiKey);
    $tracker = new UsageTracker();
    $logger = new ConsoleLogger();

    // 3. Wrap the Provider
    // The MiddlewareStack pattern intercepts API calls, enabling logging, tracking, or custom caching.
    echo "Wrapping raw OpenAIProvider with Logging and Cost Tracking middlewares...\n\n";
    $provider = MiddlewareStack::wrap($rawProvider)
        ->with(new LoggingMiddleware($logger))
        ->with(new CostTrackingMiddleware($tracker));

    // --- REQUEST 1: Basic Text Generation ---
    echo "--- REQUEST 1: Basic Text ---\n";
    $response = LLMesh::generateText(
        $provider,
        GenerateTextOptions::make()->withPrompt('Tell me a 1-line joke about computers.')
    );
    echo "\nJoke Result: " . $response->getText() . "\n\n";

    // --- REQUEST 2: Streaming Generation ---
    echo "--- REQUEST 2: Streaming Text ---\n";
    $stream = LLMesh::streamText(
        $provider,
        GenerateTextOptions::make()->withPrompt('Say "Observability is key" in German.')
    );
    
    echo "Stream output: ";
    foreach ($stream as $chunk) {
        echo $chunk->text;
        flush();
    }
    echo "\n\n";

    // 4. Inspect Usage & Cost Summary
    // The CostTrackingMiddleware updates the UsageTracker automatically.
    $summary = $tracker->getSummary();
    echo "=== Usage & Cost Summary ===\n";
    echo "Total Calls:      " . $summary['calls'] . "\n";
    echo "Input Tokens:     " . $summary['tokens_in'] . "\n";
    echo "Output Tokens:    " . $summary['tokens_out'] . "\n";
    echo "Total Tokens:     " . $summary['total_tokens'] . "\n";
    echo "Estimated Cost:   $" . number_format($summary['cost_usd'], 6) . " USD\n";
    echo "============================\n";

} catch (\Throwable $e) {
    echo "❌ Error occurred: " . $e->getMessage() . "\n";
}
