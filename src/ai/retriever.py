import os
from typing import List
from concurrent.futures import ThreadPoolExecutor

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# =========================================================
# 공통 설정
# =========================================================

# 실제 Qdrant 적재 시 사용한 컬렉션명과 반드시 같아야 함
COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME",
    "IoT_하드웨어_보안_통합점검"
)


# =========================================================
# Parent Document Retriever
# =========================================================

class ParentDocumentRetriever:
    """
    Child Chunk 검색 후 해당 Chunk가 속한
    원본 페이지(Parent Document)를 반환하는 Retriever
    """

    def __init__(
        self,
        client: QdrantClient,
        embeddings: OpenAIEmbeddings
    ):
        self.client = client
        self.embeddings = embeddings

        self.child_collection = COLLECTION_NAME

        # Child VectorStore
        self.child_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.child_collection,
            embedding=self.embeddings
        )

        print("Parent docstore 생성 중...")

        self.parent_docstore = (
            self._build_parent_docstore()
        )

        print(
            "Parent docstore 생성 완료: "
            f"{len(self.parent_docstore)}개 페이지"
        )


    def _make_parent_key(
        self,
        metadata: dict
    ) -> str:
        """
        서로 다른 PDF의 page_1, page_2가
        충돌하지 않도록 source + parent_id 사용

        예:
        ESP32-H2_기술_참조_매뉴얼.pdf::page_216
        IoT_공통보안가이드.pdf::page_20
        """

        source = metadata.get(
            "source",
            "unknown"
        )

        parent_id = metadata.get(
            "parent_id",
            "unknown"
        )

        return f"{source}::{parent_id}"


    def _build_parent_docstore(
        self
    ) -> dict:
        """
        Qdrant의 모든 Child Chunk를 가져와
        source + parent_id 기준으로 결합하여
        Parent Document Store 생성

        Returns:
            {
                "source::parent_id": Document
            }
        """

        try:

            # -----------------------------------------
            # 1. Qdrant의 모든 Child Chunk 가져오기
            # -----------------------------------------

            all_chunks = []

            offset = None

            while True:

                results = self.client.scroll(
                    collection_name=self.child_collection,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                points, next_offset = results

                if not points:
                    break

                for point in points:

                    payload = point.payload or {}

                    all_chunks.append(
                        Document(
                            page_content=payload.get(
                                "page_content",
                                ""
                            ),
                            metadata=payload.get(
                                "metadata",
                                {}
                            )
                        )
                    )

                if next_offset is None:
                    break

                offset = next_offset


            # -----------------------------------------
            # 2. Parent별 Child Chunk 그룹화
            # -----------------------------------------

            parent_groups = {}

            for chunk in all_chunks:

                metadata = chunk.metadata

                parent_id = metadata.get(
                    "parent_id"
                )

                # parent_id 없는 데이터 제외
                if not parent_id:
                    continue

                parent_key = (
                    self._make_parent_key(
                        metadata
                    )
                )

                if parent_key not in parent_groups:
                    parent_groups[parent_key] = []

                parent_groups[parent_key].append(
                    chunk
                )


            # -----------------------------------------
            # 3. Parent Document 생성
            # -----------------------------------------

            parent_docstore = {}

            for parent_key, chunks in (
                parent_groups.items()
            ):

                # chunk_index가 존재하면
                # 원래 순서대로 정렬
                chunks = sorted(
                    chunks,
                    key=lambda x: x.metadata.get(
                        "chunk_index",
                        0
                    )
                )

                combined_content = "\n\n".join(
                    chunk.page_content
                    for chunk in chunks
                    if chunk.page_content
                )

                first_chunk = chunks[0]

                parent_doc = Document(
                    page_content=combined_content,
                    metadata={
                        "source": (
                            first_chunk.metadata.get(
                                "source",
                                "알 수 없음"
                            )
                        ),
                        "page": (
                            first_chunk.metadata.get(
                                "page"
                            )
                        ),
                        "parent_id": (
                            first_chunk.metadata.get(
                                "parent_id"
                            )
                        ),
                        "category": (
                            first_chunk.metadata.get(
                                "category",
                                ""
                            )
                        )
                    }
                )

                parent_docstore[
                    parent_key
                ] = parent_doc

            return parent_docstore


        except Exception as e:

            print(
                f"Parent docstore 생성 오류: {e}"
            )

            return {}


    def search(
        self,
        query: str,
        k: int = 2
    ) -> List[Document]:
        """
        Child Chunk 검색
        → source + parent_id 추출
        → Parent Document 반환
        """

        try:

            # -----------------------------------------
            # 1. Child Chunk 검색
            # -----------------------------------------

            child_results = (
                self.child_vectorstore
                .similarity_search(
                    query,
                    k=k * 3
                )
            )

            if not child_results:
                return []


            # -----------------------------------------
            # 2. Parent Key 추출
            # -----------------------------------------

            parent_keys = []

            for doc in child_results:

                parent_id = doc.metadata.get(
                    "parent_id"
                )

                if not parent_id:
                    continue

                parent_key = (
                    self._make_parent_key(
                        doc.metadata
                    )
                )

                if (
                    parent_key
                    not in parent_keys
                ):
                    parent_keys.append(
                        parent_key
                    )

                if len(parent_keys) >= k:
                    break


            # -----------------------------------------
            # 3. Parent Document 반환
            # -----------------------------------------

            parent_docs = []

            for parent_key in parent_keys:

                if (
                    parent_key
                    in self.parent_docstore
                ):
                    parent_docs.append(
                        self.parent_docstore[
                            parent_key
                        ]
                    )

            return parent_docs


        except Exception as e:

            print(
                "Parent Document Retriever "
                f"오류: {e}"
            )

            return []


# =========================================================
# Metadata Filtered Retriever
# =========================================================

class MetadataFilteredRetriever:
    """
    IoT 하드웨어·보안 카테고리를 이용하여
    Qdrant 메타데이터 필터링 검색
    """

    def __init__(
        self,
        client: QdrantClient,
        embeddings: OpenAIEmbeddings
    ):
        self.client = client
        self.embeddings = embeddings

        self.collection_name = COLLECTION_NAME

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )


    def search(
        self,
        query: str,
        k: int = 3,
        categories: List[str] = None
    ) -> List[Document]:
        """
        IoT 카테고리 기반 메타데이터 검색

        가능한 category:

        - 회로_전원_신호설계
        - PCB_배선_기판설계
        - MCU_메모리_부품설계
        - 인터페이스_통신설계
        - 하드웨어_물리보안
        - 인증_접근통제
        - 암호화_데이터보호
        - 펌웨어_플랫폼보안
        """

        try:

            filter_conditions = None


            # -----------------------------------------
            # category 필터 생성
            # -----------------------------------------

            if categories:

                # 하나의 category
                if len(categories) == 1:

                    filter_conditions = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.category",
                                match=models.MatchValue(
                                    value=categories[0]
                                )
                            )
                        ]
                    )

                # 여러 category → OR
                else:

                    filter_conditions = models.Filter(
                        should=[
                            models.FieldCondition(
                                key="metadata.category",
                                match=models.MatchValue(
                                    value=category
                                )
                            )
                            for category in categories
                        ]
                    )


            # -----------------------------------------
            # 검색 수행
            # -----------------------------------------

            results = (
                self.vectorstore
                .similarity_search(
                    query,
                    k=k,
                    filter=filter_conditions
                )
            )


            # -----------------------------------------
            # 필터 결과가 없으면 전체 검색
            # -----------------------------------------

            if (
                not results
                and filter_conditions is not None
            ):

                print(
                    "  카테고리 필터 결과 없음 "
                    "→ 필터 없이 재검색"
                )

                results = (
                    self.vectorstore
                    .similarity_search(
                        query,
                        k=k
                    )
                )

            return results


        except Exception as e:

            print(
                "Metadata Filtered Retriever "
                f"오류: {e}"
            )

            return []


# =========================================================
# Vector Retriever
# =========================================================

class VectorRetriever:
    """
    ParentDocumentRetriever
    +
    MetadataFilteredRetriever

    두 검색기를 병렬 실행하여
    IoT 문서 검색 결과 통합
    """

    def __init__(self):
        """Qdrant 벡터 검색기 초기화"""

        # -----------------------------------------
        # Qdrant Client
        # -----------------------------------------

        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv(
                "QDRANT_API_KEY"
            )
        )


        # -----------------------------------------
        # Embedding Model
        # -----------------------------------------

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )


        # -----------------------------------------
        # Retriever 초기화
        # -----------------------------------------

        self.parent_retriever = (
            ParentDocumentRetriever(
                self.client,
                self.embeddings
            )
        )

        self.metadata_retriever = (
            MetadataFilteredRetriever(
                self.client,
                self.embeddings
            )
        )


    def search(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.5,
        categories: List[str] = None
    ) -> List[Document]:
        """
        IoT 기술문서 병렬 벡터 검색

        Parent Document 검색
        +
        Metadata Filter 검색

        결과를 합친 후 중복 제거
        """

        try:

            print(
                f"[Retriever] query={query}"
            )

            print(
                "[Retriever] categories="
                f"{categories}"
            )


            # -----------------------------------------
            # 1. 두 Retriever 병렬 실행
            # -----------------------------------------

            with ThreadPoolExecutor(
                max_workers=2
            ) as executor:

                parent_future = (
                    executor.submit(
                        self.parent_retriever.search,
                        query,
                        max(1, k // 2)
                    )
                )

                metadata_future = (
                    executor.submit(
                        self.metadata_retriever.search,
                        query,
                        k,
                        categories
                    )
                )

                parent_results = (
                    parent_future.result()
                )

                metadata_results = (
                    metadata_future.result()
                )


            # -----------------------------------------
            # 2. 결과 통합 + 중복 제거
            # -----------------------------------------

            all_results = []

            seen_docs = set()


            def add_document(
                doc: Document
            ):
                """
                source + page + content 일부를 이용하여
                중복 문서 제거
                """

                source = doc.metadata.get(
                    "source",
                    ""
                )

                page = doc.metadata.get(
                    "page",
                    ""
                )

                content_preview = (
                    doc.page_content[:200]
                )

                document_key = (
                    source,
                    page,
                    content_preview
                )

                if document_key not in seen_docs:

                    seen_docs.add(
                        document_key
                    )

                    all_results.append(
                        doc
                    )


            # Parent 먼저 추가
            for doc in parent_results:
                add_document(doc)

            # Metadata 결과 추가
            for doc in metadata_results:
                add_document(doc)


            # -----------------------------------------
            # 3. 최대 결과 개수 제한
            # -----------------------------------------

            final_results = (
                all_results[:k * 2]
            )

            print(
                "[Retriever] 검색 완료: "
                f"{len(final_results)}건"
            )

            return final_results


        except Exception as e:

            print(
                f"벡터 검색 오류: {e}"
            )


            # -----------------------------------------
            # 실패 시 Metadata 검색이라도 수행
            # -----------------------------------------

            try:

                return (
                    self.metadata_retriever
                    .search(
                        query,
                        k=k,
                        categories=categories
                    )
                )

            except Exception:

                return []


    def is_relevant(
        self,
        results: List[Document],
        min_count: int = 1
    ) -> bool:
        """
        검색 결과가 최소 개수 이상인지 확인
        """

        return (
            len(results) >= min_count
        )


# =========================================================
# Retriever Factory
# =========================================================

def get_retriever() -> VectorRetriever:
    """
    IoT 하드웨어·보안 통합
    VectorRetriever 인스턴스 반환
    """

    return VectorRetriever()