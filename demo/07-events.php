<?php

/**
 * LLMesh Demo 07: PSR-14 Event Dispatching (OpenAI)
 *
 * This script demonstrates how to configure and use a PSR-14 event dispatcher
 * with LLMesh. By listening to lifecycle events (like GenerationStarted and
 * GenerationCompleted), you can easily integrate auditing, custom tracing,
 * metrics collectors, or UI loaders into your application.
 */

// Load the bootstrap file to handle autoloading and API key setup.
$apiKey = require __DIR__ . '/bootstrap.php';

use LLMesh\Core\LLMesh;
use LLMesh\Core\Generators\GenerateTextOptions;
use LLMesh\OpenAI\OpenAIProvider;
use Psr\EventDispatcher\EventDispatcherInterface;
use LLMesh\Core\Events\GenerationStarted;
use LLMesh\Core\Events\GenerationCompleted;

echo "=== LLMesh PSR-14 Event Dispatcher Demo ===\n\n";

// 1. Define a Basic Event Dispatcher
// LLMesh conforms to the PSR-14 standard. You can pass any PSR-14 event dispatcher
// (e.g. Symfony EventDispatcher, Laravel Dispatcher, or a custom one).
// Here we define a simple dispatcher class for demonstration.
class SimpleEventDispatcher implements EventDispatcherInterface
{
    private array $listeners = [];

    // Registers a callback for a specific event class type
    public function addListener(string $eventClass, callable $listener): void
    {
        $this->listeners[$eventClass][] = $listener;
    }

    // Standard PSR-14 dispatch method
    public function dispatch(object $event): object
    {
        $eventClass = get_class($event);
        if (isset($this->listeners[$eventClass])) {
            foreach ($this->listeners[$eventClass] as $listener) {
                $listener($event);
            }
        }
        return $event;
    }
}

try {
    $provider = new OpenAIProvider($apiKey);
    
    // 2. Instantiate and Configure the Dispatcher
    $dispatcher = new SimpleEventDispatcher();

    // Event 1: Generation Started
    $dispatcher->addListener(GenerationStarted::class, function (GenerationStarted $event) {
        echo "\n📢 [Event Triggered] GenerationStarted:\n";
        echo "   - Provider: " . $event->provider . "\n";
        echo "   - Prompt:   " . ($event->options->prompt ?? '(No prompt)') . "\n";
        echo "------------------------------------------------\n";
    });

    // Event 2: Generation Completed
    $dispatcher->addListener(GenerationCompleted::class, function (GenerationCompleted $event) {
        echo "\n📢 [Event Triggered] GenerationCompleted:\n";
        echo "   - Provider:    " . $event->provider . "\n";
        echo "   - Duration:    " . $event->durationMs . " ms\n";
        echo "   - Token Usage: Input=" . $event->response->getUsage()->getInputTokens() . 
                           ", Output=" . $event->response->getUsage()->getOutputTokens() . 
                           ", Total=" . $event->response->getUsage()->getTotalTokens() . "\n";
        echo "------------------------------------------------\n";
    });

    // 3. Initialize LLMesh with the Dispatcher
    // Instead of using the static facade, we build an instance configured with the dispatcher.
    $llmesh = LLMesh::make()->withEventDispatcher($dispatcher);

    // 4. Run Generation
    $prompt = 'Say "OpenAI event dispatching works!" in French.';
    echo "Sending prompt: \"{$prompt}\"...\n";

    $response = $llmesh->generateText(
        $provider,
        GenerateTextOptions::make()->withPrompt($prompt)
    );

    echo "\nFinal Output: " . $response->getText() . "\n\n";

} catch (\Throwable $e) {
    echo "❌ Error occurred: " . $e->getMessage() . "\n";
}
