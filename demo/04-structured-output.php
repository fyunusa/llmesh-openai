<?php

/**
 * LLMesh Demo 04: Structured Output (OpenAI)
 *
 * This script demonstrates how to force OpenAI GPT-4o to return structured JSON data
 * that conforms to a specific schema you define. This is highly useful for data extraction,
 * API integrations, and ensuring consistent output formats.
 */

// Load the bootstrap file to handle autoloading and API key setup.
$apiKey = require __DIR__ . '/bootstrap.php';

use LLMesh\Core\LLMesh;
use LLMesh\Core\Generators\GenerateObjectOptions;
use LLMesh\Core\Schema\Schema;
use LLMesh\OpenAI\OpenAIProvider;

echo "=== LLMesh OpenAI Structured Output Demo ===\n\n";

try {
    // 1. Initialize the Provider
    $provider = new OpenAIProvider($apiKey);

    // 2. Define the Target Schema
    // We want the AI to extract a person's name, age, and interests.
    // We define this structure using the LLMesh Schema builder.
    echo "Defining schema for the target object...\n";
    $schema = Schema::object([
        'name'      => Schema::string()->required()->description("The person's full name"),
        'age'       => Schema::integer()->required()->minimum(0),
        'interests' => Schema::array(Schema::string())->required()->description("A list of hobbies or interests"),
    ])->required(['name', 'age', 'interests']);

    // 3. Define the Raw Input Text
    $inputText = "Alice is a 28-year-old medical doctor who enjoys hiking in the mountains, photography, and oil painting.";
    echo "Input Text: \"{$inputText}\"\n\n";

    // 4. Configure Options
    // We pass both the prompt and the schema we want to enforce.
    $options = GenerateObjectOptions::make()
        ->withPrompt("Extract information from: {$inputText}")
        ->withSchema($schema);

    // 5. Generate the Object
    // Instead of generateText, we use generateObject.
    echo "Requesting structured extraction from OpenAI...\n";
    $response = LLMesh::generateObject($provider, $options);

    // 6. Inspect the Parsed Result
    // LLMesh automatically parses and validates the JSON response into a PHP stdClass object.
    echo "\n=== Structured Output Succeeded ===\n";
    echo "Parsed Object Structure:\n";
    print_r($response->object);

    // Access specific fields:
    echo "\nAccessing individual properties:\n";
    echo " - Name:      " . $response->object['name'] . "\n";
    echo " - Age:       " . $response->object['age'] . "\n";
    echo " - Interests: " . implode(', ', $response->object['interests']) . "\n";
    echo "=====================================\n";

} catch (\LLMesh\Core\Exceptions\ValidationException $e) {
    // If the model fails to return output matching the schema, a ValidationException is thrown.
    echo "❌ Validation Error: The response did not match the required schema!\n";
    print_r($e->errors());
} catch (\Throwable $e) {
    echo "❌ Error occurred: " . $e->getMessage() . "\n";
}
