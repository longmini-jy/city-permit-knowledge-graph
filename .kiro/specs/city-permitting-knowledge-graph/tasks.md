# Implementation Plan

## Vertical Slicing Strategy

Each slice delivers end-to-end value with minimal scope, allowing for fast iteration and learning.

**Slice 1 (Week 1-2)**: Single document → Single query → Manual extraction
**Slice 2 (Week 3-4)**: Add LLM extraction → Basic evaluation
**Slice 3 (Week 5-6)**: Add vector search → Hybrid retrieval
**Slice 4 (Week 7-8)**: Scale to 10 documents → Evaluation pipeline
**Slice 5 (Week 9-10)**: Add versioning → Monitoring → Deploy

## Phase 1: Oakland MVP - Vertical Slices (Weeks 1-10)

### Slice 1: End-to-End Proof of Concept (Week 1-2)
**Goal**: Single document → Single query → Working answer (manual everything)
**Deliverable**: Can query "What permits do I need for a deck in Oakland?" and get an answer

- [ ] 1. Minimal setup and first document
  - [x] 1.1 Create Python project with FastAPI
    - Initialize repo, venv, basic dependencies
    - Set up OpenAI API key
    - _Requirements: All (foundation)_
  
  - [ ] 1.2 Manually process ONE Oakland permit document
    - Download one residential building permit PDF
    - Manually extract text and save as JSON
    - Create simple data structure: {permit_type, requirements[], fees, timeline}
    - _Requirements: 1.1, 3.1, 3.2, 3.4_
  
  - [ ] 1.3 Create single API endpoint
    - POST /query with question parameter
    - Hardcode: return the manual JSON for any query
    - Test with curl
    - _Requirements: 5.1, 5.2_
  
  - [ ] 1.4 Add simple LLM call
    - Send question + manual JSON to GPT-4
    - Ask LLM to answer based on context
    - Return LLM response
    - _Requirements: 3.1, 5.4_

**Success Criteria**: Can ask a question and get an answer. Proves the concept works.

### Slice 2: Add Real LLM Extraction (Week 3-4)
**Goal**: Replace manual JSON with LLM extraction from PDF text
**Deliverable**: Automated extraction from ONE document

- [ ] 2. Automate extraction with LLM
  - [ ] 2.1 Add PDF text extraction
    - Use pdfplumber to extract text from PDF
    - Save raw text
    - _Requirements: 1.1_
  
  - [ ] 2.2 Build LLM extraction prompt
    - Create prompt: "Extract permit info from this text: {text}"
    - Define JSON schema for structured output
    - Use GPT-4 function calling
    - _Requirements: 3.1, 3.2, 3.4_
  
  - [ ] 2.3 Validate extraction quality
    - Compare LLM output to manual JSON
    - Calculate accuracy (% fields correct)
    - Log token costs
    - _Requirements: 7.1_
  
  - [ ]* 2.4 Write basic test
    - Test extraction on sample document
    - Verify schema compliance
    - _Requirements: 3.1_

**Success Criteria**: LLM extracts permit info automatically. Know the cost and accuracy.

### Slice 3: Add Vector Search (Week 5-6)
**Goal**: Scale to 5 documents with semantic search
**Deliverable**: Can find relevant document chunks for any query

- [ ] 3. Implement chunking and vector search
  - [ ] 3.1 Add 4 more Oakland documents
    - Download 4 more permit PDFs (different types)
    - Extract text from all 5 documents
    - _Requirements: 1.1_
  
  - [ ] 3.2 Implement chunking
    - Split each document into 512-token chunks with 50-token overlap
    - Store chunks with metadata (doc_id, chunk_id, permit_type)
    - _Requirements: 1.1_
  
  - [ ] 3.3 Set up Chroma vector DB
    - Install chromadb
    - Generate embeddings for all chunks (OpenAI)
    - Store in Chroma with metadata
    - _Requirements: 1.1, 1.2_
  
  - [ ] 3.4 Implement vector search
    - Embed user query
    - Search Chroma for top-5 similar chunks
    - Pass chunks to LLM for extraction
    - _Requirements: 5.1, 5.2_
  
  - [ ]* 3.5 Test retrieval quality
    - Create 5 test queries
    - Manually judge if top-5 chunks are relevant
    - Calculate Precision@5
    - _Requirements: 7.1_

**Success Criteria**: Can query across 5 documents. Retrieval works. Know retrieval quality.

### Slice 4: Scale to 10 Documents + Evaluation (Week 7-8)
**Goal**: Scale to 10 documents, add proper evaluation
**Deliverable**: Production-quality evaluation pipeline

- [ ] 4. Scale and evaluate
  - [ ] 4.1 Add 5 more documents (total 10)
    - Process through chunking + embedding pipeline
    - Store in Chroma
    - _Requirements: 1.1, 1.2_
  
  - [ ] 4.2 Create test set
    - Write 20 test queries with ground truth answers
    - Label which documents are relevant for each query
    - _Requirements: 7.1, 7.3_
  
  - [ ] 4.3 Build evaluation pipeline
    - Calculate Precision@k, Recall@k, MRR
    - Calculate extraction accuracy (% fields correct)
    - Track costs per query
    - _Requirements: 7.1_
  
  - [ ] 4.4 Add metadata filtering
    - Filter by permit_type before vector search
    - Compare filtered vs unfiltered retrieval
    - _Requirements: 5.1, 5.2_
  
  - [ ]* 4.5 Write property test for retrieval
    - **Property 21: Address-based jurisdiction filtering**
    - **Validates: Requirements 5.1**

**Success Criteria**: 10 documents working. Know exact quality metrics. Can iterate to improve.

### Slice 5: Add Versioning + Monitoring + Deploy (Week 9-10)
**Goal**: Production-ready with versioning and monitoring
**Deliverable**: Deployed API with monitoring dashboard

- [ ] 5. Production readiness
  - [ ] 5.1 Implement dataset versioning
    - Create version v1 with current 10 documents
    - Store metadata (documents, embeddings, extraction model, metrics)
    - Save as Parquet
    - _Requirements: 4.1, 4.2_
  
  - [ ] 5.2 Add comprehensive logging
    - Log all LLM calls (prompt, response, cost, latency)
    - Log all queries (question, retrieved chunks, answer)
    - Log errors and retries
    - _Requirements: 7.2_
  
  - [ ] 5.3 Build monitoring dashboard
    - Display: daily costs, query volume, retrieval quality, extraction accuracy
    - Show: recent queries, errors, slow queries
    - _Requirements: 7.1, 7.2_
  
  - [ ] 5.4 Add caching
    - Cache embeddings by content hash (Redis)
    - Cache LLM responses by (chunks_hash, prompt)
    - Track cache hit rates
    - _Requirements: 5.1, 5.2_
  
  - [ ] 5.5 Deploy to production
    - Deploy to Railway/Render
    - Set up PostgreSQL + Redis
    - Configure environment variables
    - Test in production
    - _Requirements: All_
  
  - [ ] 5.6 Create simple web UI
    - Single page: search box + results
    - Show answer, sources, confidence
    - _Requirements: 5.1, 5.2_

**Success Criteria**: Live API. Can demo to potential customers. Have metrics to show quality.

### Slice 6: Horizontal Scaling (Week 11-12, Optional)
**Goal**: Scale to 20+ documents, optimize costs
**Deliverable**: Cost-optimized, higher quality system

- [ ] 6. Optimize and scale
  - [ ] 6.1 Add 10 more documents (total 20)
    - Process through pipeline
    - Create version v2
    - _Requirements: 1.1, 1.2_
  
  - [ ] 6.2 Implement hybrid retrieval
    - Combine vector search + keyword matching
    - Implement reranking
    - Compare strategies on test set
    - _Requirements: 5.1, 5.2, 5.5_
  
  - [ ] 6.3 Optimize costs
    - Use cheaper embedding model for some queries
    - Implement smarter caching
    - Batch LLM calls where possible
    - _Requirements: 3.1, 5.1_
  
  - [ ] 6.4 Add entity resolution
    - Normalize permit types to taxonomy
    - Deduplicate requirements across documents
    - _Requirements: 2.1, 3.1_
  
  - [ ]* 6.5 Write property tests for normalization
    - **Property 6: Permit type taxonomy mapping**
    - **Property 12: Requirement parsing completeness**
    - **Validates: Requirements 2.1, 3.2**

**Success Criteria**: 20 documents. Lower cost per query. Higher quality. Ready to onboard first customer.

## Phase 2: Customer Validation and Expansion (Future, Week 13+)

**After validating with customers, prioritize based on feedback:**

- [ ] 7. Expand to SF and Berkeley (if customers want multi-city)
  - Adapt pipeline for format variations
  - Process 10 documents per city
  - Update evaluation for multi-city
  - _Requirements: 1.1, 1.2, 2.1, 2.2_

- [ ] 8. Add dependency analysis (if customers need permit sequences)
  - Extract permit dependencies from documents
  - Build dependency graph
  - Add to API responses
  - _Requirements: 9.1, 9.2, 9.3_

- [ ] 9. Add timeline estimation (if customers need planning)
  - Collect historical data
  - Build simple statistical model
  - Add timeline endpoint
  - _Requirements: 6.1, 6.2, 6.3_

- [ ] 10. Build automated ingestion (when scaling to many cities)
  - Implement web scraping
  - Add scheduled jobs
  - Implement change detection
  - _Requirements: 1.1, 1.2, 7.4_

- [ ] 11. Add city administrator tools (if cities want to validate)
  - Build validation interface
  - Add correction workflow
  - _Requirements: 10.1, 10.2, 10.3_
