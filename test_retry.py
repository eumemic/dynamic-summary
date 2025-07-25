#!/usr/bin/env python3
"""Quick test to demonstrate retry counter functionality."""

import asyncio
import logging
from ragzoom.config import RagZoomConfig
from ragzoom.store import Store
from ragzoom.index import TreeBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_retry_counter():
    # Create a test document with dense content that will likely need retries
    dense_text = """
    In the year 1878 I took my degree of Doctor of Medicine of the University of London, and proceeded to Netley to go through the course prescribed for surgeons in the army. Having completed my studies there, I was duly attached to the Fifth Northumberland Fusiliers as Assistant Surgeon. The regiment was stationed in India at the time, and before I could join it, the second Afghan war had broken out. On landing at Bombay, I learned that my corps had advanced through the passes, and was already deep in the enemy's country. I followed, however, with many other officers who were in the same situation as myself, and succeeded in reaching Candahar in safety, where I found my regiment, and at once entered upon my new duties.
    
    The campaign brought honours and promotion to many, but for me it had nothing but misfortune and disaster. I was removed from my brigade and attached to the Berkshires, with whom I served at the fatal battle of Maiwand. There I was struck on the shoulder by a Jezail bullet, which shattered the bone and grazed the subclavian artery. I should have fallen into the hands of the murderous Ghazis had it not been for the devotion and courage shown by Murray, my orderly, who threw me across a pack-horse, and succeeded in bringing me safely to the British lines.
    """ * 5  # Repeat to make it dense
    
    # Configure with small token limits to force retries
    config = RagZoomConfig(
        leaf_tokens=300,
        adjacent_context_tokens=200,
        openai_api_key="sk-test",  # Will need real key
    )
    
    # Create store and builder
    store = Store(config)
    builder = TreeBuilder(config, store)
    
    # Test document
    print("Testing retry counter with dense text...")
    try:
        doc_id = await builder.add_document_async(dense_text, "test_doc", show_progress=False)
        print(f"Document indexed: {doc_id}")
    except Exception as e:
        print(f"Error (expected if no API key): {e}")

if __name__ == "__main__":
    asyncio.run(test_retry_counter())