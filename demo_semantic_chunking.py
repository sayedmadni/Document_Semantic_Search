#!/usr/bin/env python3
"""
Semantic Chunking Demo

This script demonstrates the difference between recursive chunking and semantic chunking
for patent documents and other technical content.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_chunking_methods():
    """Demonstrate different chunking methods"""
    
    print("=== Semantic vs Recursive Chunking Demo ===\n")
    
    # Sample patent-like text
    sample_text = """
    A robotic system for household assistance comprising a humanoid robot with artificial intelligence capabilities. 
    The robot includes a central processing unit configured to process sensor data from multiple cameras and sensors. 
    Machine learning algorithms analyze environmental data to make navigation decisions. The system utilizes deep learning 
    neural networks for object recognition and manipulation tasks. Computer vision algorithms enable the robot to 
    identify household objects and perform complex manipulation tasks. The robotic system includes safety mechanisms 
    to prevent collisions and ensure human safety during operation. Advanced path planning algorithms optimize movement 
    efficiency while avoiding obstacles in dynamic environments.
    
    The artificial intelligence system processes natural language commands from users through speech recognition. 
    Natural language processing modules interpret user intentions and translate them into robotic actions. 
    The system maintains a knowledge base of household tasks and procedures. Reinforcement learning algorithms 
    enable the robot to improve performance through experience and user feedback. The robotic system integrates 
    with smart home devices and IoT sensors for comprehensive environmental awareness.
    """
    
    try:
        from chonkie import RecursiveChunker, SemanticChunker
        
        print("📄 Sample Text:")
        print(sample_text.strip())
        print("\n" + "="*80 + "\n")
        
        # Recursive Chunking
        print("🔧 RECURSIVE CHUNKING:")
        recursive_chunker = RecursiveChunker()
        recursive_chunks = recursive_chunker(sample_text)
        
        for i, chunk in enumerate(recursive_chunks, 1):
            print(f"Chunk {i} ({len(chunk.text.split())} words):")
            print(f"  {chunk.text.strip()}")
            print()
        
        print("="*80 + "\n")
        
        # Semantic Chunking
        print("🧠 SEMANTIC CHUNKING:")
        semantic_chunker = SemanticChunker(
            embedding_model="all-MiniLM-L6-v2",
            threshold=0.8,
            chunk_size=2048,
            similarity_window=3,
            min_sentences_per_chunk=2,
            min_characters_per_sentence=24
        )
        semantic_chunks = semantic_chunker(sample_text)
        
        for i, chunk in enumerate(semantic_chunks, 1):
            print(f"Chunk {i} ({len(chunk.text.split())} words):")
            print(f"  {chunk.text.strip()}")
            print()
        
        print("="*80 + "\n")
        
        # Comparison
        print("📊 COMPARISON:")
        print(f"Recursive chunks: {len(recursive_chunks)}")
        print(f"Semantic chunks: {len(semantic_chunks)}")
        
        # Analyze chunk characteristics
        recursive_word_counts = [len(chunk.text.split()) for chunk in recursive_chunks]
        semantic_word_counts = [len(chunk.text.split()) for chunk in semantic_chunks]
        
        print(f"\nRecursive chunk sizes: {recursive_word_counts}")
        print(f"Semantic chunk sizes: {semantic_word_counts}")
        
        print(f"\nRecursive avg chunk size: {sum(recursive_word_counts) / len(recursive_word_counts):.1f} words")
        print(f"Semantic avg chunk size: {sum(semantic_word_counts) / len(semantic_word_counts):.1f} words")
        
        # Show semantic coherence
        print("\n🎯 SEMANTIC COHERENCE ANALYSIS:")
        print("Recursive chunking splits text based on structural patterns (sentences, paragraphs)")
        print("Semantic chunking groups related concepts together based on meaning similarity")
        
        print("\n✅ ADVANTAGES OF SEMANTIC CHUNKING FOR PATENTS:")
        print("• Preserves conceptual relationships across sentence boundaries")
        print("• Better for technical documents with complex terminology")
        print("• Maintains context for patent claims and descriptions")
        print("• Improves retrieval accuracy for semantic search")
        print("• Reduces information fragmentation")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure chonkie is installed: pip install chonkie")
    except Exception as e:
        print(f"❌ Error during demo: {e}")

def demo_patent_specific_chunking():
    """Demonstrate semantic chunking with patent-specific content"""
    
    print("\n=== Patent-Specific Semantic Chunking Demo ===\n")
    
    patent_text = """
    Field of the Invention: This invention relates to humanoid robotics and artificial intelligence systems 
    for household assistance applications.
    
    Background of the Invention: Traditional household robots lack the dexterity and intelligence required 
    for complex manipulation tasks. Existing systems rely on pre-programmed routines and cannot adapt 
    to dynamic environments or learn from user interactions.
    
    Summary of the Invention: The present invention provides a humanoid robotic system with advanced 
    artificial intelligence capabilities. The system includes machine learning algorithms for object 
    recognition, natural language processing for user interaction, and computer vision for environmental 
    understanding. The robot can perform complex household tasks such as cooking, cleaning, and organizing 
    items through intelligent manipulation and planning algorithms.
    
    Detailed Description: The robotic system comprises a humanoid body structure with articulated joints 
    and sensors. The central processing unit runs deep learning models for perception and decision-making. 
    Computer vision algorithms process input from multiple cameras to identify objects and obstacles. 
    The system uses reinforcement learning to improve task performance over time through user feedback 
    and experience accumulation.
    """
    
    try:
        from chonkie import SemanticChunker
        
        print("📋 Patent Document Sample:")
        print(patent_text.strip())
        print("\n" + "="*80 + "\n")
        
        # Configure semantic chunker for patent documents
        patent_chunker = SemanticChunker(
            embedding_model="all-MiniLM-L6-v2",
            threshold=0.75,  # Lower threshold for more cohesive chunks
            chunk_size=1536,  # Smaller chunks for better precision
            similarity_window=2,
            min_sentences_per_chunk=1,
            min_characters_per_sentence=20
        )
        
        chunks = patent_chunker(patent_text)
        
        print("🧠 SEMANTIC CHUNKS FOR PATENT DOCUMENT:")
        for i, chunk in enumerate(chunks, 1):
            words = len(chunk.text.split())
            print(f"\nChunk {i} ({words} words):")
            print(f"  {chunk.text.strip()}")
            
            # Analyze chunk content
            if "field" in chunk.text.lower() or "invention" in chunk.text.lower():
                print("  🏷️  Contains: Field/Background info")
            elif "summary" in chunk.text.lower() or "present invention" in chunk.text.lower():
                print("  📝 Contains: Summary/Overview")
            elif "detailed" in chunk.text.lower() or "comprises" in chunk.text.lower():
                print("  🔧 Contains: Technical details")
        
        print(f"\n📊 Total chunks: {len(chunks)}")
        print(f"📊 Average chunk size: {sum(len(chunk.text.split()) for chunk in chunks) / len(chunks):.1f} words")
        
    except Exception as e:
        print(f"❌ Error in patent demo: {e}")

def main():
    """Run the semantic chunking demonstration"""
    print("🚀 Starting Semantic Chunking Demonstration\n")
    
    # Basic chunking comparison
    demo_chunking_methods()
    
    # Patent-specific demonstration
    demo_patent_specific_chunking()
    
    print("\n" + "="*80)
    print("🎉 Demo completed!")
    print("\n💡 Key Takeaways:")
    print("• Semantic chunking preserves meaning across sentence boundaries")
    print("• Better for technical documents like patents")
    print("• Improves search accuracy and context preservation")
    print("• Recommended for your patent semantic search system")

if __name__ == "__main__":
    main()
