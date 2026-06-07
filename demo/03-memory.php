<?php

/**
 * LLMesh Demo 03: Conversation Memory (OpenAI)
 *
 * This script demonstrates how to add conversational memory to your LLMesh queries.
 * By using memory, the LLM can remember context and details from previous interactions
 * in the same session, enabling natural back-and-forth chat.
 */

// Load the bootstrap file to handle autoloading and API key setup.
$apiKey = require __DIR__ . '/bootstrap.php';

use LLMesh\Core\LLMesh;
use LLMesh\Core\Generators\GenerateTextOptions;
use LLMesh\Core\Memory\InMemoryStore;
use LLMesh\OpenAI\OpenAIProvider;

echo "=== LLMesh OpenAI Conversation Memory Demo ===\n\n";

try {
    // 1. Initialize the Provider
    $provider = new OpenAIProvider($apiKey);

    // 2. Initialize the Memory Store
    // LLMesh provides an InMemoryStore to save messages in memory during the execution lifetime.
    // In a production app, you might use a database-backed or Redis-backed store.
    echo "Initializing Memory Store...\n";
    $store = new InMemoryStore();

    // 3. Define a Session ID
    // The Session ID uniquely identifies this specific chat conversation.
    // Multiple users or chat windows would each have their own unique Session ID.
    $sessionId = 'openai-user-session-123';
    echo "Session ID: {$sessionId}\n\n";

    // --- ROUND 1: Introduce Ourselves ---
    echo "Round 1: Sending message containing user information...\n";
    $prompt1 = "Hello! My name is Charlie and my favorite food is pepperoni pizza.";
    echo "Prompt 1: \"{$prompt1}\"\n";

    // We configure the memory store and session ID in our options.
    $options1 = GenerateTextOptions::make()
        ->withPrompt($prompt1)
        ->withMemory($store, $sessionId);

    $response1 = LLMesh::generateText($provider, $options1);
    echo "OpenAI response:\n";
    echo "------------------------------------------------\n";
    echo $response1->getText() . "\n";
    echo "------------------------------------------------\n\n";


    // --- ROUND 2: Test Memory Recall ---
    // In this round, we don't mention our name or food again.
    // The LLM must retrieve this information from the memory store.
    echo "Round 2: Asking a follow-up question that relies on memory...\n";
    $prompt2 = "What is my name and what is my favorite food?";
    echo "Prompt 2: \"{$prompt2}\"\n";

    $options2 = GenerateTextOptions::make()
        ->withPrompt($prompt2)
        ->withMemory($store, $sessionId); // Pass the same store and session ID

    $response2 = LLMesh::generateText($provider, $options2);
    echo "OpenAI response:\n";
    echo "------------------------------------------------\n";
    echo $response2->getText() . "\n";
    echo "------------------------------------------------\n\n";

    echo "Conversation memory worked successfully!\n";

} catch (\Throwable $e) {
    echo "❌ Error occurred: " . $e->getMessage() . "\n";
}
