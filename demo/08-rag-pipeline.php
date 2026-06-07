<?php

/**
 * LLMesh Demo 08: RAG Ingestion & Semantic Retrieval (OpenAI)
 *
 * This script demonstrates the LLMesh Retrieval-Augmented Generation (RAG) system.
 * We load raw text data, split it into chunks, generate vector embeddings using
 * OpenAI's embedding model, store them in a vector store, and perform a semantic search
 * to retrieve the most relevant chunks.
 */

// Load the bootstrap file to handle autoloading and API key setup.
$apiKey = require __DIR__ . '/bootstrap.php';

use LLMesh\Core\RAG\Pipeline;
use LLMesh\Core\RAG\Loaders\ArrayLoader;
use LLMesh\Core\RAG\Splitters\RecursiveCharacterSplitter;
use LLMesh\Core\RAG\VectorStores\InMemoryVectorStore;
use LLMesh\OpenAI\OpenAIProvider;

echo "=== LLMesh OpenAI RAG Pipeline Demo ===\n\n";

try {
    // 1. Prepare Sample Source Documents
    // In a real application, these could be database rows, PDF contents, or webpages.
    $texts = [
        'LLMesh is a lightweight PHP SDK designed for building robust AI applications, text generators, multi-step agents, and observability pipelines.',
        'Retrieval-Augmented Generation (RAG) splits text documents, embeds them into vectors, and retrieves top matching chunks for context-aware queries.',
        'While Python has LangChain, PHP developers use LLMesh to build modular multi-provider AI application setups easily.',
        'Organic gardening tips: plant seed potatoes in full sun using well-drained, acidic soil rich in organic compost.',
    ];

    // Optional metadata to filter or tag each chunk
    $metadata = [
        ['category' => 'tech', 'source' => 'llmesh-docs'],
        ['category' => 'rag', 'source' => 'rag-wiki'],
        ['category' => 'tech', 'source' => 'blog-post'],
        ['category' => 'gardening', 'source' => 'gardener-guide'],
    ];

    // 2. Set Up RAG Pipeline Components
    echo "Initializing RAG Components...\n";
    // Loaders wrap the source data.
    $loader = new ArrayLoader($texts, $metadata);

    // Splitters break long documents into manageable overlapping chunks.
    $splitter = new RecursiveCharacterSplitter(chunkSize: 100, overlap: 10);

    // OpenAIProvider generates the embeddings (uses text-embedding-3-small under the hood).
    $provider = new OpenAIProvider($apiKey);

    // VectorStore retains the high-dimensional vectors and retrieves them semantically.
    $vectorStore = new InMemoryVectorStore();

    // 3. Configure the Pipeline
    $pipeline = Pipeline::make()
        ->load($loader)
        ->split($splitter)
        ->embed($provider)
        ->store($vectorStore)
        // Set a progress listener callback
        ->onProgress(function (int $done, int $total) {
            echo "   [Ingestion Progress] Processed chunk {$done}/{$total}\n";
        });

    // 4. Run the Ingestion Pipeline
    // This loads, splits, calls OpenAI's embedding API, and stores the vectors.
    echo "Running Ingestion Pipeline...\n";
    $result = $pipeline->run();

    echo "\nIn-Memory Vector Database Ingested Successfully!\n";
    echo " - Documents Loaded:  " . $result->documentsLoaded . "\n";
    echo " - Chunks Created:    " . $result->chunksCreated . "\n";
    echo " - Chunks Stored:     " . $result->chunksStored . "\n";
    echo " - Ingestion Time:    " . $result->durationMs . " ms\n\n";

    // 5. Query & Retrieve Relevant Context
    // We execute a natural language search. The pipeline embeds the search query
    // and returns the top-K chunks with the highest cosine similarity.
    $query = "Tell me about LLMesh and RAG in PHP";
    echo "Searching Vector Store for: \"{$query}\"...\n";

    // Retrieve the top 2 matching document chunks
    $retrievedDocs = $pipeline->retrieve($query, topK: 2);

    echo "\n=== Top Semantic Search Matches ===\n";
    foreach ($retrievedDocs as $index => $doc) {
        echo "\nRank " . ($index + 1) . " (Document Chunk ID: {$doc->id}):\n";
        echo "   Content:  \"" . trim($doc->content) . "\"\n";
        echo "   Metadata: " . json_encode($doc->metadata) . "\n";
    }
    echo "=====================================\n";

} catch (\Throwable $e) {
    echo "❌ Error occurred: " . $e->getMessage() . "\n";
}
