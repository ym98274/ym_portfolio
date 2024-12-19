import chromadb
import time
import logging

logging.basicConfig(level=logging.INFO)  # Set to DEBUG for detailed logs
logger = logging.getLogger(__name__)


def initialize_in_memory_client():
    """
    Initialize an in-memory ChromaDB client.
    This client will not persist data to disk and avoids dependency on file storage.

    Returns:
        chromadb.Client: Initialized in-memory client.
    """
    logger.info("Initializing ChromaDB client...")
    return chromadb.Client()


def retry_on_failure(func, retries=3, delay=2, *args, **kwargs):
    """
    Retry a function on failure.
    Args:
        func (callable): The function to retry.
        retries (int): Number of retry attempts.
        delay (int): Delay between retries (in seconds).
        *args, **kwargs: Arguments to pass to the function.
    Returns:
        Any: Result of the function if successful.
    Raises:
        Exception: If all retries fail.
    """
    for attempt in range(retries):
        try:
            logger.debug(f"Attempt {attempt + 1} for function {func.__name__}")
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


def create_database(client, name):
    """
    Create or get a ChromaDB collection.

    Args:
        client (chromadb.Client): ChromaDB client instance.
        name (str): Name of the collection.

    Returns:
        chromadb.api.Collection: The created or retrieved ChromaDB collection.
    """
    logger.info(f"Creating or retrieving collection: {name}")
    return retry_on_failure(client.get_or_create_collection, name=name)


def add_frames_to_database(collection, captions, frame_indexes):
    """
    Add frames with captions and indexes to a ChromaDB collection.

    Args:
        collection (chromadb.api.Collection): The ChromaDB collection.
        captions (list of str): Captions for each frame.
        frame_indexes (list of int): Indexes of the frames.
    """
    ids = [str(idx) for idx in frame_indexes]
    logger.info(f"Adding {len(ids)} frames to the collection.")
    retry_on_failure(collection.add, documents=captions, ids=ids)


def query_database(client, collection_name, query_text):
    """
    Query a ChromaDB collection for the most relevant document.

    Args:
        client (chromadb.Client): ChromaDB client instance.
        collection_name (str): Name of the collection to query.
        query_text (str): The query text.

    Returns:
        dict: Query result containing the relevant document and associated metadata.
    """
    logger.info(f"Querying collection '{collection_name}' with text: {query_text}")
    collection = retry_on_failure(client.get_collection, name=collection_name)
    results = retry_on_failure(collection.query, query_texts=[query_text], n_results=1)

    if "ids" in results and results["ids"]:
        return {
            "id": results["ids"][0],
            "distance": results["distances"][0],
        }
    else:
        return {"id": None, "distance": None}
