import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class RetrievalResult:
    """Represents a chunk retrieval result"""
    chunk_id: str
    content: str
    source: str
    page: int
    side: str
    similarity_score: float
    sentence_query: str
    database_name: str  # Added to track which database the chunk came from

@dataclass
class SentenceResults:
    """Results for a single sentence query"""
    sentence: str
    top_chunks: List[RetrievalResult]
    sentence_index: int

class MultiDatabaseSentenceRAGSystem:
    """
    RAG system that can search across multiple chunk databases simultaneously
    """
    
    def __init__(self, chunks_files: Union[str, List[str]] = None, 
                 chunks_data_dict: Dict[str, List[Dict]] = None):
        """
        Initialize the multi-database RAG system
        
        Args:
            chunks_files: Single file path, list of file paths, or directory containing JSON files
            chunks_data_dict: Dictionary mapping database names to chunks data
        """
        self.databases = {}  # Maps database_name -> {chunks, chunk_texts, vectorizer, vectors}
        self.all_chunk_texts = []
        self.all_chunks_with_db = []  # List of (chunk, database_name) tuples
        self.global_vectorizer = None
        self.global_vectors = None
        
        if chunks_files:
            self._load_from_files(chunks_files)
        elif chunks_data_dict:
            self._load_from_data_dict(chunks_data_dict)
        else:
            raise ValueError("Must provide either chunks_files or chunks_data_dict")
        
        self._build_global_index()
    
    def _load_from_files(self, chunks_files: Union[str, List[str]]):
        """Load chunks from file(s)"""
        file_paths = []
        
        if isinstance(chunks_files, str):
            if os.path.isdir(chunks_files):
                # Load all JSON files from directory
                file_paths = list(Path(chunks_files).glob("*.json"))
            else:
                # Single file
                file_paths = [chunks_files]
        else:
            # List of files
            file_paths = chunks_files
        
        for file_path in file_paths:
            file_path = Path(file_path)
            db_name = file_path.stem  # Use filename without extension as database name
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                self.databases[db_name] = {
                    'chunks': chunks,
                    'chunk_texts': [chunk['text'] for chunk in chunks],
                    'file_path': str(file_path)
                }
                
                # Add to global collections
                for chunk in chunks:
                    self.all_chunks_with_db.append((chunk, db_name))
                    self.all_chunk_texts.append(chunk['text'])
                
                print(f"Loaded {len(chunks)} chunks from {file_path} as database '{db_name}'")
                
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue
    
    def _load_from_data_dict(self, chunks_data_dict: Dict[str, List[Dict]]):
        """Load chunks from data dictionary"""
        for db_name, chunks in chunks_data_dict.items():
            self.databases[db_name] = {
                'chunks': chunks,
                'chunk_texts': [chunk['text'] for chunk in chunks]
            }
            
            # Add to global collections
            for chunk in chunks:
                self.all_chunks_with_db.append((chunk, db_name))
                self.all_chunk_texts.append(chunk['text'])
            
            print(f"Loaded {len(chunks)} chunks for database '{db_name}'")
    
    def _build_global_index(self):
        """Build global TF-IDF index across all databases"""
        if not self.all_chunk_texts:
            raise ValueError("No chunks loaded!")
        
        print(f"Building global TF-IDF index across {len(self.databases)} databases...")
        print(f"Total chunks: {len(self.all_chunk_texts)}")
        
        # Initialize TF-IDF vectorizer
        self.global_vectorizer = TfidfVectorizer(
            max_features=10000,         # Increased for multiple databases
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Fit and transform all chunk texts
        self.global_vectors = self.global_vectorizer.fit_transform(self.all_chunk_texts)
        print(f"Built global index with {self.global_vectors.shape[1]} features")
        
        # Print database summary
        print("\nDatabase Summary:")
        for db_name, db_info in self.databases.items():
            print(f"  - {db_name}: {len(db_info['chunks'])} chunks")
    
    def break_into_sentences(self, text: str) -> List[str]:
        """Break text into sentences using NLTK"""
        sentences = sent_tokenize(text)
        
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def search_sentence(self, sentence: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search for top-k most similar chunks across all databases"""
        # Transform sentence using the global vectorizer
        sentence_vector = self.global_vectorizer.transform([sentence])
        
        # Calculate cosine similarity with all chunks
        similarities = cosine_similarity(sentence_vector, self.global_vectors).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Create results
        results = []
        for idx in top_indices:
            chunk, db_name = self.all_chunks_with_db[idx]
            similarity_score = similarities[idx]
            
            result = RetrievalResult(
                chunk_id=chunk['id'],
                content=chunk['text'],
                source=chunk['source'],
                page=chunk['page'],
                side=chunk['side'],
                similarity_score=similarity_score,
                sentence_query=sentence,
                database_name=db_name
            )
            results.append(result)
        
        return results
    
    def search_by_database(self, sentence: str, database_name: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search within a specific database only"""
        if database_name not in self.databases:
            raise ValueError(f"Database '{database_name}' not found. Available: {list(self.databases.keys())}")
        
        db_info = self.databases[database_name]
        
        # Build temporary vectorizer for this database if not exists
        if 'vectorizer' not in db_info:
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                lowercase=True,
                strip_accents='unicode'
            )
            vectors = vectorizer.fit_transform(db_info['chunk_texts'])
            db_info['vectorizer'] = vectorizer
            db_info['vectors'] = vectors
        
        # Search within this database
        sentence_vector = db_info['vectorizer'].transform([sentence])
        similarities = cosine_similarity(sentence_vector, db_info['vectors']).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = db_info['chunks'][idx]
            similarity_score = similarities[idx]
            
            result = RetrievalResult(
                chunk_id=chunk['id'],
                content=chunk['text'],
                source=chunk['source'],
                page=chunk['page'],
                side=chunk['side'],
                similarity_score=similarity_score,
                sentence_query=sentence,
                database_name=database_name
            )
            results.append(result)
        
        return results
    
    # NEW METHOD: Search using entire prompt without sentence breakdown
    def search_whole_prompt(self, prompt: str, top_k: int = 10, search_mode: str = 'global') -> Dict:
        """
        NEW METHOD: Search using the entire prompt as a single query
        
        Args:
            prompt: Input prompt/question (treated as single query)
            top_k: Number of top chunks to retrieve
            search_mode: 'global' to search across all databases, or specific database name
            
        Returns:
            Dictionary containing search results for the whole prompt
        """
        print(f"Searching with whole prompt: {prompt[:100]}...")
        print(f"Search mode: {search_mode}")
        
        # Search the entire prompt as one query
        if search_mode == 'global':
            # Use global search across all databases
            prompt_vector = self.global_vectorizer.transform([prompt])
            similarities = cosine_similarity(prompt_vector, self.global_vectors).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Create results
            top_chunks = []
            for idx in top_indices:
                chunk, db_name = self.all_chunks_with_db[idx]
                similarity_score = similarities[idx]
                
                result = RetrievalResult(
                    chunk_id=chunk['id'],
                    content=chunk['text'],
                    source=chunk['source'],
                    page=chunk['page'],
                    side=chunk['side'],
                    similarity_score=similarity_score,
                    sentence_query=prompt,  # Store the whole prompt as the query
                    database_name=db_name
                )
                top_chunks.append(result)
        else:
            # Search within specific database
            if search_mode not in self.databases:
                raise ValueError(f"Database '{search_mode}' not found. Available: {list(self.databases.keys())}")
            
            db_info = self.databases[search_mode]
            
            # Build temporary vectorizer for this database if not exists
            if 'vectorizer' not in db_info:
                vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    lowercase=True,
                    strip_accents='unicode'
                )
                vectors = vectorizer.fit_transform(db_info['chunk_texts'])
                db_info['vectorizer'] = vectorizer
                db_info['vectors'] = vectors
            
            # Search within this database
            prompt_vector = db_info['vectorizer'].transform([prompt])
            similarities = cosine_similarity(prompt_vector, db_info['vectors']).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            top_chunks = []
            for idx in top_indices:
                chunk = db_info['chunks'][idx]
                similarity_score = similarities[idx]
                
                result = RetrievalResult(
                    chunk_id=chunk['id'],
                    content=chunk['text'],
                    source=chunk['source'],
                    page=chunk['page'],
                    side=chunk['side'],
                    similarity_score=similarity_score,
                    sentence_query=prompt,
                    database_name=search_mode
                )
                top_chunks.append(result)
        
        return {
            'original_prompt': prompt,
            'search_type': 'whole_prompt',
            'top_chunks': top_chunks,
            'total_chunks_retrieved': len(top_chunks),
            'search_mode': search_mode,
            'databases_used': list(self.databases.keys()) if search_mode == 'global' else [search_mode]
        }
    
    def query_per_sentence_topk(self, prompt: str, top_k_per_sentence: int = 5, 
                               search_mode: str = 'global') -> Dict:
        """
        NEW METHOD: Query that returns top-k chunks for each sentence without global deduplication
        
        Args:
            prompt: Input prompt/question
            top_k_per_sentence: Number of top chunks to retrieve per sentence
            search_mode: 'global' to search across all databases, or specific database name
            
        Returns:
            Dictionary containing results for each sentence with all top-k chunks preserved
        """
        # Break prompt into sentences
        sentences = self.break_into_sentences(prompt)
        
        # Search each sentence
        sentence_results = []
        all_chunks_flat = []  # All chunks from all sentences (with duplicates preserved)
        
        for i, sentence in enumerate(sentences):
            # Get top chunks for this sentence
            if search_mode == 'global':
                top_chunks = self.search_sentence(sentence, top_k_per_sentence)
            else:
                top_chunks = self.search_by_database(sentence, search_mode, top_k_per_sentence)
            
            # Create sentence result
            sentence_result = SentenceResults(
                sentence=sentence,
                top_chunks=top_chunks,
                sentence_index=i
            )
            sentence_results.append(sentence_result)
            
            # Add all chunks to flat list (preserving duplicates)
            all_chunks_flat.extend(top_chunks)
        
        return {
            'original_prompt': prompt,
            'search_type': 'per_sentence',
            'sentences': sentences,
            'sentence_results': sentence_results,
            'all_chunks_flat': all_chunks_flat,  # All chunks with duplicates preserved
            'total_sentences': len(sentences),
            'total_chunks_retrieved': len(all_chunks_flat),
            'search_mode': search_mode,
            'databases_used': list(self.databases.keys()) if search_mode == 'global' else [search_mode]
        }
    
    def query(self, prompt: str, top_k_per_sentence: int = 5, 
              search_mode: str = 'global') -> Dict:
        """
        Main query method (ORIGINAL - with global deduplication)
        
        Args:
            prompt: Input prompt/question
            top_k_per_sentence: Number of top chunks to retrieve per sentence
            search_mode: 'global' to search across all databases, or specific database name
            
        Returns:
            Dictionary containing results for each sentence
        """
        # Break prompt into sentences
        sentences = self.break_into_sentences(prompt)
        
        # Search each sentence
        sentence_results = []
        all_chunks = []
        
        for i, sentence in enumerate(sentences):
            # Get top chunks for this sentence
            if search_mode == 'global':
                top_chunks = self.search_sentence(sentence, top_k_per_sentence)
            else:
                top_chunks = self.search_by_database(sentence, search_mode, top_k_per_sentence)
            
            # Create sentence result
            sentence_result = SentenceResults(
                sentence=sentence,
                top_chunks=top_chunks,
                sentence_index=i
            )
            sentence_results.append(sentence_result)
            
            # Collect all chunks for overall ranking
            all_chunks.extend(top_chunks)
        
        # Get unique chunks across all sentences
        unique_chunks = self._deduplicate_chunks(all_chunks)
        
        return {
            'original_prompt': prompt,
            'search_type': 'sentence_based_deduplicated',
            'sentences': sentences,
            'sentence_results': sentence_results,
            'all_retrieved_chunks': unique_chunks,
            'total_sentences': len(sentences),
            'total_unique_chunks': len(unique_chunks),
            'search_mode': search_mode,
            'databases_used': list(self.databases.keys()) if search_mode == 'global' else [search_mode]
        }
    
    def _deduplicate_chunks(self, chunks: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicate chunks and keep the one with highest similarity"""
        chunk_dict = {}
        
        for chunk in chunks:
            # Use chunk_id + database_name as unique key
            chunk_key = f"{chunk.database_name}:{chunk.chunk_id}"
            if chunk_key not in chunk_dict or chunk.similarity_score > chunk_dict[chunk_key].similarity_score:
                chunk_dict[chunk_key] = chunk
        
        # Sort by similarity score (descending)
        unique_chunks = list(chunk_dict.values())
        unique_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return unique_chunks
    
    def print_results(self, results: Dict, show_content: bool = True, max_content_length: int = 500):
        """Print search results in a readable format"""
        print("\n" + "="*80)
        print("MULTI-DATABASE RAG SEARCH RESULTS")
        print("="*80)
        
        print(f"\nOriginal Prompt: {results['original_prompt']}")
        print(f"Search Type: {results.get('search_type', 'unknown')}")
        print(f"Search Mode: {results['search_mode']}")
        print(f"Databases Used: {', '.join(results['databases_used'])}")
        
        # Handle different result types
        if results.get('search_type') == 'whole_prompt':
            # Whole prompt search results
            print(f"Total Chunks Retrieved: {results['total_chunks_retrieved']}")
            
            print(f"\n{'='*60}")
            print("TOP CHUNKS FOR WHOLE PROMPT")
            print(f"{'='*60}")
            
            for i, chunk in enumerate(results['top_chunks'], 1):
                print(f"\n{i}. [{chunk.database_name}] {chunk.source} (Score: {chunk.similarity_score:.4f})")
                print(f"   Chunk ID: {chunk.chunk_id}")
                if show_content:
                    content = chunk.content[:max_content_length]
                    if len(chunk.content) > max_content_length:
                        content += "..."
                    print(f"   Content: {content}")
        
        else:
            # Sentence-based search results
            if 'total_sentences' in results:
                print(f"Total Sentences: {results['total_sentences']}")
            
            # Check if this is per-sentence results or deduplicated results
            if 'all_chunks_flat' in results:
                print(f"Total Chunks Retrieved: {results['total_chunks_retrieved']} (with duplicates)")
            elif 'total_unique_chunks' in results:
                print(f"Total Unique Chunks Retrieved: {results['total_unique_chunks']}")
            
            # Results per sentence (if available)
            if 'sentence_results' in results:
                for sentence_result in results['sentence_results']:
                    print(f"\n{'-'*60}")
                    print(f"SENTENCE {sentence_result.sentence_index + 1}: {sentence_result.sentence}")
                    print(f"{'-'*60}")
                    
                    for j, chunk in enumerate(sentence_result.top_chunks, 1):
                        print(f"\n  Rank {j}: [{chunk.database_name}] {chunk.source} (Score: {chunk.similarity_score:.4f})")
                        print(f"    Chunk ID: {chunk.chunk_id}")
                        
                        if show_content:
                            content = chunk.content[:max_content_length]
                            if len(chunk.content) > max_content_length:
                                content += "..."
                            print(f"    Content: {content}")
            
            # Overall top chunks section (if available)
            if 'all_retrieved_chunks' in results:
                print(f"\n{'='*60}")
                print("TOP UNIQUE CHUNKS ACROSS ALL SENTENCES")
                print(f"{'='*60}")
                
                for i, chunk in enumerate(results['all_retrieved_chunks'][:10], 1):
                    print(f"\n{i}. [{chunk.database_name}] {chunk.source} (Score: {chunk.similarity_score:.4f})")
                    print(f"   Chunk ID: {chunk.chunk_id}")
                    if show_content:
                        content = chunk.content[:max_content_length]
                        if len(chunk.content) > max_content_length:
                            content += "..."
                        print(f"   Content: {content}")
    
    def get_context_for_generation_per_sentence(self, results: Dict) -> str:
        """
        Create context string from per-sentence results (preserves all chunks)
        
        Args:
            results: Results from query_per_sentence_topk method
            
        Returns:
            Context string with all chunks organized by sentence
        """
        context_parts = []
        
        for sentence_result in results['sentence_results']:
            context_parts.append(f"\n--- Sentence {sentence_result.sentence_index + 1}: {sentence_result.sentence} ---")
            
            for i, chunk in enumerate(sentence_result.top_chunks, 1):
                context_parts.append(f"\n[Chunk {i} - {chunk.database_name}:{chunk.source}]")
                context_parts.append(chunk.content)
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_context_for_generation_flat(self, results: Dict) -> str:
        """
        Create context string from all chunks in flat list (preserves duplicates)
        
        Args:
            results: Results from query_per_sentence_topk method
            
        Returns:
            Flat context string with all chunks
        """
        context_parts = []
        
        chunks_to_use = results.get('all_chunks_flat', results.get('top_chunks', []))
        
        for i, chunk in enumerate(chunks_to_use, 1):
            context_parts.append(f"\n[Chunk {i} - {chunk.database_name}:{chunk.source}]")
            context_parts.append(f"Query: {chunk.sentence_query}")
            context_parts.append(chunk.content)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_context_for_generation(self, results: Dict, max_chunks: int = 10) -> str:
        """Create a context string from top chunks for text generation"""
        if 'top_chunks' in results:
            # Whole prompt search results
            top_chunks = results['top_chunks'][:max_chunks]
        elif 'all_retrieved_chunks' in results:
            # Original method with deduplication
            top_chunks = results['all_retrieved_chunks'][:max_chunks]
        else:
            # Per-sentence method - take first max_chunks from flat list
            top_chunks = results.get('all_chunks_flat', [])[:max_chunks]
        
        context_parts = []
        
        for i, chunk in enumerate(top_chunks, 1):
            context_parts.append(f"\n[Chunk {i} - {chunk.database_name}:{chunk.source}]")
            context_parts.append(chunk.content)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_database_stats(self) -> Dict:
        """Get statistics about loaded databases"""
        stats = {
            'total_databases': len(self.databases),
            'total_chunks': len(self.all_chunk_texts),
            'databases': {}
        }
        
        for db_name, db_info in self.databases.items():
            stats['databases'][db_name] = {
                'chunk_count': len(db_info['chunks']),
                'file_path': db_info.get('file_path', 'N/A')
            }
        
        return stats

# Convenience functions for easy usage
def create_multi_rag_system(chunks_files: Union[str, List[str]]) -> MultiDatabaseSentenceRAGSystem:
    """Create a multi-database RAG system from chunk files"""
    return MultiDatabaseSentenceRAGSystem(chunks_files=chunks_files)

def search_across_databases(rag_system: MultiDatabaseSentenceRAGSystem, prompt: str, 
                           top_k: int = 5, search_mode: str = 'global', 
                           print_results: bool = True) -> Dict:
    """
    Search a prompt across multiple databases (ORIGINAL - with deduplication)
    
    Args:
        rag_system: Initialized multi-database RAG system
        prompt: Query prompt
        top_k: Top chunks per sentence
        search_mode: 'global' for all databases or specific database name
        print_results: Whether to print formatted results
        
    Returns:
        Search results dictionary
    """
    results = rag_system.query(prompt, top_k_per_sentence=top_k, search_mode=search_mode)
    
    if print_results:
        rag_system.print_results(results)
    
    return results

def search_per_sentence_topk(rag_system: MultiDatabaseSentenceRAGSystem, prompt: str, 
                            top_k: int = 5, search_mode: str = 'global', 
                            print_results: bool = False) -> Dict:
    """
    Search a prompt and get top-k chunks per sentence (no global deduplication)
    
    Args:
        rag_system: Initialized multi-database RAG system
        prompt: Query prompt
        top_k: Top chunks per sentence
        search_mode: 'global' for all databases or specific database name
        print_results: Whether to print formatted results
        
    Returns:
        Search results dictionary with all chunks preserved
    """
    results = rag_system.query_per_sentence_topk(prompt, top_k_per_sentence=top_k, search_mode=search_mode)
    
    if print_results:
        rag_system.print_results(results)
    
    return results

# NEW FUNCTION: Search using the whole prompt
def search_whole_prompt(rag_system: MultiDatabaseSentenceRAGSystem, prompt: str, 
                       top_k: int = 10, search_mode: str = 'global', 
                       print_results: bool = False) -> Dict:
    """
    NEW FUNCTION: Search using the entire prompt as a single query (no sentence breakdown)
    
    Args:
        rag_system: Initialized multi-database RAG system
        prompt: Query prompt (treated as single query)
        top_k: Number of top chunks to retrieve
        search_mode: 'global' for all databases or specific database name
        print_results: Whether to print formatted results
        
    Returns:
        Search results dictionary
    """
    results = rag_system.search_whole_prompt(prompt, top_k=top_k, search_mode=search_mode)
    
    if print_results:
        rag_system.print_results(results)
    
    return results

# Example usage:
"""
# Create RAG system
rag_system = create_multi_rag_system(['chunks1.json', 'chunks2.json'])

# NEW: Search using whole prompt (no sentence breakdown)
results = search_whole_prompt(rag_system, "Your complete query here", top_k=10)

# Get context for generation from whole prompt search
context = rag_system.get_context_for_generation(results)
"""