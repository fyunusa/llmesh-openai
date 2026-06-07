<?php

/**
 * LLMesh Demo 05: Agents and Tools (OpenAI Function Calling)
 *
 * This script demonstrates how to build an Agent that has access to custom Tools.
 * The model (GPT-4o) decides when it needs to call a tool, outputs the parameters,
 * LLMesh executes the local PHP function (the tool handler), returns the result
 * to the model, and the model generates a final, informed answer.
 */

// Load the bootstrap file to handle autoloading and API key setup.
$apiKey = require __DIR__ . '/bootstrap.php';

use LLMesh\Core\Agents\Agent;
use LLMesh\Core\Tools\Tool;
use LLMesh\OpenAI\OpenAIProvider;

echo "=== LLMesh OpenAI Agent & Tools Demo ===\n\n";

// 1. Define a Custom Tool
// Tools represent functions that the LLM can decide to execute.
// Here, we define a tool to calculate the cube of a number.
echo "Registering 'calculate_cube' tool...\n";
$calculatorTool = Tool::make('calculate_cube')
    ->description('Calculate the cube of a given integer. Use this whenever the user asks for the cube of a number.')
    ->parameters([
        'number' => Tool::integer('The integer to raise to the power of 3')->required(),
    ])
    ->handler(function (array $params): array {
        // This is a local PHP callback. When the LLM decides to use the tool,
        // LLMesh runs this PHP code.
        $num = $params['number'];
        $result = $num * $num * $num;
        echo "   [Local Tool Execution] Cubing {$num} -> Result: {$result}\n";
        return ['result' => $result];
    });

try {
    // 2. Initialize the Provider
    $provider = new OpenAIProvider($apiKey);

    // 3. Create the Agent
    // We supply the provider, system instructions, tools, and maximum run steps to avoid infinite loops.
    $agent = Agent::make(
        provider:     $provider,
        systemPrompt: 'You are a helpful math assistant. Use the calculate_cube tool when asked to find the cube of a number. Explain your answer clearly.',
        tools:        [$calculatorTool],
        maxSteps:     5 // Limit agent execution to a max of 5 round-trips
    )
    // The onStep hook runs after each step (LLM generation or Tool call).
    // This allows us to observe the agent's internal reasoning loop.
    ->onStep(function ($step) {
        echo "--- Agent Step Completed ---\n";
        if (!empty($step->toolCalls)) {
            foreach ($step->toolCalls as $call) {
                echo "🤖 Model decided to call tool: '{$call->name}'\n";
                echo "   Arguments: " . json_encode($call->arguments) . "\n";
            }
        } else {
            echo "🤖 Model returned the final response.\n";
        }
    });

    // 4. Run the Agent
    $userQuestion = 'What is the cube of 6?';
    echo "Asking Agent: \"{$userQuestion}\"\n\n";

    $result = $agent->run($userQuestion);

    // 5. Output the final answer
    echo "\n================ Final Answer ================\n";
    echo $result->finalText . "\n";
    echo "==============================================\n";

} catch (\Throwable $e) {
    echo "❌ Error occurred during agent run: " . $e->getMessage() . "\n";
}
