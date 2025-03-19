import fitz  # PyMuPDF
import argparse
from openai import OpenAI, BadRequestError
from openai.types import CreateEmbeddingResponse
import numpy as np
import logging
import uuid
import time

# from openai.types import
import os
from qdrant_client import QdrantClient
from qdrant_client.models import UpdateResult, PointStruct, VectorParams, Distance
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1024

assert (
    QDRANT_URL and QDRANT_API_KEY and OPENAI_API_KEY
), "Please provide QDRANT_URL, QDRANT_API_KEY and OPENAI_API_KEY in .env file"


oai_client = OpenAI(api_key=OPENAI_API_KEY)

qd_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def upsert_random_data(length: int) -> UpdateResult:
    points: list[PointStruct] = [
        PointStruct(
            id=uuid.uuid4().hex,
            vector=np.random.rand(VECTOR_SIZE).tolist(),
            payload={"rand": str(np.random.rand(1))},
        )
        for i in range(length)
    ]

    result = qd_client.upsert(
        collection_name="test",
        points=points,
    )
    return result


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    text.replace("\n", " ")

    log.info(f"Extracted text from {pdf_path}")
    log.info(f"Text length: {len(text)}")
    return text


def chunk_file(
    file_path: str,
    chunk_size: int = VECTOR_SIZE,
    overlap: int = int(VECTOR_SIZE / 2),
) -> list[str]:
    text = extract_text_from_pdf(file_path)
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        max_end = min(i + chunk_size, len(text))
        chunk = text[i:max_end]
        chunks.append(chunk)
        print(".", end="")
    print()

    log.info(f"Chunked file {file_path} into {len(chunks)} chunks")
    return chunks


def create_embeddings(chunks: list[str]) -> list[PointStruct]:
    embeddings = []
    for enum, chunk in enumerate(chunks):
        log.info(f"Creating embedding for chunk {enum}/{len(chunks)}")
        log.info(f"Chunk length: {len(chunk)}")

        try:
            response: CreateEmbeddingResponse = oai_client.embeddings.create(
                input=str(chunk),
                model=EMBEDDING_MODEL,
                dimensions=VECTOR_SIZE,
            )
        except BadRequestError as e:
            log.error(f"Failed to create embedding for chunk {enum}/{len(chunks)}")
            log.error(e)
            log.error(f"Chunk: {chunk}")
            continue
        embedding = PointStruct(
            id=uuid.uuid4().hex,
            vector=response.data[0].embedding,
            payload={"text": chunk},
        )

        log.warning(f"Embedding: {sum(response.data[0].embedding)}")

        embeddings.append(embedding)

    log.info(f"Created {len(embeddings)} embeddings")
    return embeddings


def upsert_embeddings(embeddings: list[PointStruct]) -> UpdateResult:
    log.info(f"Upserting {len(embeddings)} embeddings")

    for i in range(0, len(embeddings), 100):
        log.info(f"Upserting {i}/{len(embeddings)} embeddings")
        maxlen = min(i + 100, len(embeddings))

        try:
            result = qd_client.upsert(
                collection_name="test",
                points=embeddings[i:maxlen],
            )
        except Exception as e:
            log.error(f"Failed to upsert embeddings {i}/{maxlen}")
            log.error(e)
            continue

    log.info(f"Upserted {len(embeddings)} embeddings. Result: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI and Qdrant integration")
    parser.add_argument(
        "-u",
        "--upsert",
        type=int,
        help="Insert random data to Qdrant",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="Query Qdrant for similar vectors",
    )
    parser.add_argument(
        "-c",
        "--create",
        type=str,
        help="Create a collection",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="File with data",
    )
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="Show info",
    )
    args = parser.parse_args()

    if args.create:
        result = qd_client.create_collection(
            args.create,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        log.info(f"Collection created: {result}")

    elif args.upsert:
        for i in range(10):
            result = upsert_random_data(1000)
            log.info(f"Upserted {args.upsert} vectors. Result: {result}")

    elif args.query:
        embeddings = create_embeddings([args.query])

        tstart = time.time()
        result = qd_client.search(
            collection_name="test",
            query_vector=embeddings[0].vector,
            limit=3,
        )
        tend = time.time()
        for res in result:
            log.info(f"Result: {res}")
        log.info(f"Search time: {tend - tstart}")

    elif args.file:
        text = extract_text_from_pdf(args.file)
        chunks = chunk_file(args.file)
        embeddings = create_embeddings(chunks)
        result = upsert_embeddings(embeddings)
        log.info(f"Upserted {len(embeddings)} vectors. Result: {result}")

    elif args.info:
        result = qd_client.get_collection("test")
        log.info(f"Info: {result}")
    else:
        parser.print_help()
