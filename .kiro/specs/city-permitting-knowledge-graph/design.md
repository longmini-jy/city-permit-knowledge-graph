# Design Document

## Overview

The City Permitting Knowledge Graph System is a data infrastructure platform that transforms disparate municipal permitting data into a standardized, queryable, versioned knowledge graph accessible via API. The system addresses the critical market gap in machine-readable permitting data by ingesting documents from multiple formats (PDFs, HTML, spreadsheets, scanned images), extracting structured entities, normalizing to a consistent schema, and serving results through a robust query interface.

The MVP will focus on Oakland, CA as the initial data source, with planned expansion to San Francisco, Berkeley, and the greater Seattle area. This phased approach allows validation of the ingestion and normalization pipeline before horizontal scaling to additional municipalities.

The system is designed as data infrastructure - not a chatbot - providing the compliance intelligence layer that real estate platforms, construction-tech companies, licensing SaaS tools, and insurance companies need to integrate permitting requirements into their products.

## Architecture

### High-Level Architecture

The system follows an ML-infra pipeline architecture with document processing, embedding generation, hybrid retrieval, and LLM-based extraction:

```
City Sources → Document Ingestion → Chunking Strategy → Embedding Generation → Vector DB Storage
                                                                                        ↓
                                                                                  Hybrid Retrieval
                                                                                  (Vector + Filters)
                                                                                        ↓
API Consumers ← Query API ← Entity Resolution ← LLM Extraction ← Retrieved Chunks
                    ↓
            Feature Serving Layer
                    ↓
            Evaluation Pipeline → Quality Metrics → Versioned Datasets
```

### Component Overview

1. **Document Ingestion Pipeline**: Fetches, parses, and chunks documents from city sources
2. **Embedding Generation**: Creates vector representations of document chunks using embedding models
3. **Vector Database**: Stores embeddings and enables semantic similarity search
4. **Hybrid Retrieval Engine**: Combines vector search with rule-based filters for optimal results
5. **LLM Extraction Service**: Uses LLM APIs with structured outputs to extract permit entities
6. **Entity Resolution Layer**: Normalizes and deduplicates extracted entities into knowledge graph
7. **Feature Serving API**: Serves permit data with versioning and caching
8. **Evaluation Pipeline**: Measures retrieval quality and extraction accuracy
9. **Dataset Versioning**: Tracks data versions for reproducibility and governance

### Technology Stack

- **LLM API**: OpenAI GPT-4 or Anthropic Claude for structured entity extraction with function calling
- **Vector Database**: Chroma (local/embedded) or Pinecone (managed) for semantic search and retrieval
- **Embedding Model**: OpenAI text-embedding-3-small or sentence-transformers for document embeddings
- **Graph Database**: Neo4j (optional for Phase 2) or PostgreSQL with graph extensions for entity relationships
- **Document Processing**: PyPDF2/pdfplumber for PDF parsing, BeautifulSoup for HTML
- **API Layer**: FastAPI (Python) with async support for LLM calls
- **Feature Store**: Simple versioned JSON/Parquet files for MVP, evolve to Feast or Tecton later
- **Caching**: Redis for LLM response caching and rate limiting
- **Monitoring**: LangSmith or custom logging for LLM call tracking, Prometheus for system metrics
- **Evaluation**: Custom evaluation pipeline for retrieval quality (precision@k, recall@k, MRR)

## ML Infrastructure Design

### Chunking Strategy

**Objective**: Split documents into semantically meaningful chunks for embedding and retrieval.

**Approach**:
- **Hierarchical chunking**: Preserve document structure (sections, subsections, paragraphs)
- **Chunk size**: 512-1024 tokens (balance between context and specificity)
- **Overlap**: 50-100 tokens between chunks to avoid boundary issues
- **Metadata preservation**: Each chunk retains document ID, section, page number, source URL

**Implementation**:
```python
class ChunkingStrategy:
    def chunk_document(self, document: Document, chunk_size: int = 512, overlap: int = 50) -> List[Chunk]:
        """Split document into overlapping chunks with metadata"""
        
    def preserve_structure(self, document: Document) -> List[Section]:
        """Identify document structure (headings, sections)"""
        
    def create_chunk_metadata(self, chunk: str, document: Document, position: int) -> ChunkMetadata:
        """Attach metadata to each chunk"""
```

### Embedding Generation

**Objective**: Convert text chunks into dense vector representations for semantic search.

**Model Selection**:
- **MVP**: OpenAI text-embedding-3-small (1536 dimensions, $0.02/1M tokens)
- **Alternative**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions, free, local)

**Pipeline**:
```python
class EmbeddingService:
    def generate_embeddings(self, chunks: List[Chunk]) -> List[Embedding]:
        """Generate embeddings for chunks with batching"""
        
    def batch_embed(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Batch embedding generation for efficiency"""
        
    def cache_embeddings(self, chunk_id: str, embedding: np.ndarray):
        """Cache embeddings to avoid recomputation"""
```

**Optimization**:
- Batch API calls (100 chunks per request)
- Cache embeddings by content hash
- Monitor token usage and costs

### Vector Database Operations

**Objective**: Store and retrieve embeddings with metadata filtering.

**Schema Design**:
```python
@dataclass
class VectorRecord:
    id: str
    embedding: np.ndarray  # 1536-dim vector
    text: str  # Original chunk text
    metadata: dict  # {city, permit_type, section, confidence, source_url, date}
```

**Operations**:
```python
class VectorStore:
    def insert(self, records: List[VectorRecord]):
        """Insert embeddings with metadata"""
        
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 10, filters: dict = None) -> List[VectorRecord]:
        """Retrieve similar chunks with optional metadata filters"""
        
    def hybrid_search(self, query: str, filters: dict, alpha: float = 0.5) -> List[VectorRecord]:
        """Combine vector similarity with keyword matching"""
```

**Indexing Strategy**:
- Use HNSW (Hierarchical Navigable Small World) for fast approximate search
- Create indexes on metadata fields (city, permit_type, date)
- Partition by city for scalability

### Hybrid Retrieval Engine

**Objective**: Combine semantic search with rule-based filters for optimal precision and recall.

**Architecture**:
```python
class HybridRetriever:
    def retrieve(self, query: str, filters: dict, strategy: str = "hybrid") -> List[Chunk]:
        """
        Strategies:
        - vector_only: Pure semantic search
        - filter_first: Apply filters then vector search
        - rerank: Vector search then rerank with filters
        - hybrid: Weighted combination of vector + keyword
        """
        
    def apply_filters(self, results: List[VectorRecord], filters: dict) -> List[VectorRecord]:
        """Filter by city, permit_type, date_range, confidence_threshold"""
        
    def rerank(self, results: List[VectorRecord], query: str) -> List[VectorRecord]:
        """Rerank results using cross-encoder or LLM"""
```

**Filter Types**:
- **Jurisdiction filters**: city, county, state
- **Permit type filters**: category, subcategory
- **Temporal filters**: effective_date, expiration_date
- **Quality filters**: confidence_score > threshold

### LLM Extraction Service

**Objective**: Use LLM APIs to extract structured permit data from retrieved chunks.

**Approach**: Function calling / structured outputs with schema validation

**Implementation**:
```python
class LLMExtractor:
    def extract_permit_info(self, chunks: List[Chunk], schema: dict) -> PermitRule:
        """Extract structured data using LLM with function calling"""
        
    def build_prompt(self, chunks: List[Chunk], extraction_type: str) -> str:
        """Construct prompt with retrieved context"""
        
    def validate_output(self, llm_response: dict, schema: dict) -> ValidationResult:
        """Validate LLM output against schema"""
        
    def handle_errors(self, error: Exception) -> RetryStrategy:
        """Handle rate limits, timeouts, invalid outputs"""
```

**Extraction Schemas**:
```python
permit_schema = {
    "permit_type": {"type": "string", "required": True},
    "requirements": {"type": "array", "items": {"type": "string"}},
    "fees": {"type": "object", "properties": {"base_amount": "number", "calculation_method": "string"}},
    "timeline": {"type": "object", "properties": {"duration": "number", "unit": "string"}},
    "dependencies": {"type": "array", "items": {"type": "string"}},
    "confidence": {"type": "number", "min": 0, "max": 1}
}
```

**Cost Optimization**:
- Cache LLM responses by (chunks_hash, schema)
- Use cheaper models for simple extractions
- Batch similar extractions
- Monitor token usage per request

### Entity Resolution Layer

**Objective**: Normalize and deduplicate extracted entities across documents and sources.

**Challenges**:
- Same permit type described differently across cities
- Duplicate requirements with slight wording variations
- Conflicting information from multiple sources

**Implementation**:
```python
class EntityResolver:
    def resolve_permit_type(self, extracted_type: str, city: str) -> StandardPermitType:
        """Map to standardized taxonomy using embeddings + rules"""
        
    def deduplicate_requirements(self, requirements: List[str]) -> List[str]:
        """Remove duplicate requirements using semantic similarity"""
        
    def resolve_conflicts(self, entities: List[Entity]) -> Entity:
        """Choose authoritative entity based on source confidence"""
        
    def build_knowledge_graph(self, entities: List[Entity]) -> Graph:
        """Construct relationships between entities"""
```

**Resolution Strategies**:
- **Embedding-based**: Compare embeddings of entity descriptions
- **Rule-based**: Exact string matching, regex patterns
- **LLM-based**: Ask LLM if two entities are equivalent
- **Hybrid**: Combine all approaches with confidence scoring

### Feature Serving API

**Objective**: Serve permit data with low latency, versioning, and caching.

**Architecture**:
```python
class FeatureServer:
    def get_permit_features(self, permit_id: str, version: str = "latest") -> PermitFeatures:
        """Retrieve permit features with version control"""
        
    def batch_get_features(self, permit_ids: List[str]) -> List[PermitFeatures]:
        """Batch feature retrieval for efficiency"""
        
    def cache_features(self, permit_id: str, features: PermitFeatures, ttl: int = 3600):
        """Cache frequently accessed features"""
```

**Versioning**:
- Each dataset version has unique ID (v1, v2, v3)
- Track schema changes across versions
- Support point-in-time queries
- Maintain backward compatibility

### Evaluation Pipeline

**Objective**: Measure and track retrieval quality and extraction accuracy.

**Metrics**:

**Retrieval Metrics**:
- **Precision@k**: % of top-k results that are relevant
- **Recall@k**: % of relevant documents in top-k results
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant result
- **NDCG (Normalized Discounted Cumulative Gain)**: Ranking quality metric

**Extraction Metrics**:
- **Field completeness**: % of required fields extracted
- **Accuracy**: % of extracted values matching ground truth
- **Confidence calibration**: Correlation between confidence scores and accuracy

**Implementation**:
```python
class EvaluationPipeline:
    def evaluate_retrieval(self, queries: List[Query], ground_truth: List[RelevanceJudgment]) -> RetrievalMetrics:
        """Measure retrieval quality"""
        
    def evaluate_extraction(self, extractions: List[Extraction], ground_truth: List[PermitRule]) -> ExtractionMetrics:
        """Measure extraction accuracy"""
        
    def create_test_set(self, documents: List[Document], sample_size: int) -> TestSet:
        """Create labeled test set for evaluation"""
        
    def track_metrics_over_time(self, metrics: Metrics, version: str):
        """Log metrics for monitoring"""
```

**Test Set Creation**:
- Manually label 50-100 permit documents
- Create query-document relevance judgments
- Validate extracted entities against source documents
- Version test sets alongside data versions

### Dataset Versioning

**Objective**: Track data versions for reproducibility, governance, and rollback.

**Versioning Strategy**:
```python
class DatasetVersioning:
    def create_version(self, dataset: Dataset, metadata: dict) -> Version:
        """Create immutable dataset version"""
        
    def get_version(self, version_id: str) -> Dataset:
        """Retrieve specific dataset version"""
        
    def compare_versions(self, v1: str, v2: str) -> VersionDiff:
        """Show differences between versions"""
        
    def rollback(self, version_id: str):
        """Rollback to previous version"""
```

**Version Metadata**:
- Version ID and timestamp
- Source documents included
- Extraction model and parameters
- Evaluation metrics
- Schema version
- Change description

**Storage**:
- Store versions as Parquet files (efficient, columnar)
- Use content-addressable storage (hash-based)
- Maintain version lineage graph
- Compress old versions

## Components and Interfaces

### 1. Document Ingestion Pipeline

**Responsibilities:**
- Discover and fetch documents from city sources
- Handle multiple formats (PDF, HTML, XLS, XLSX, CSV, scanned images)
- Manage rate limiting and respectful crawling
- Track source metadata (URL, retrieval date, content hash)

**Interfaces:**

```python
class IngestionEngine:
    def fetch_document(self, source_url: str) -> RawDocument:
        """Fetch document from city source"""
        
    def extract_text(self, raw_doc: RawDocument) -> ExtractedText:
        """Extract text content preserving structure"""
        
    def apply_ocr(self, image_doc: RawDocument) -> OCRResult:
        """Apply OCR to scanned documents with confidence scores"""
        
    def parse_html(self, html_doc: RawDocument) -> StructuredHTML:
        """Parse HTML preserving semantic structure"""
        
    def parse_spreadsheet(self, spreadsheet: RawDocument) -> TabularData:
        """Parse spreadsheet identifying headers and data"""
```

**Data Flow:**
- Input: City source URLs, document references
- Output: Extracted text with metadata, structure preservation, confidence scores

### 2. Normalization Pipeline

**Responsibilities:**
- Map permit types to standardized taxonomy
- Normalize addresses using geocoding
- Standardize fees, timelines, and measurements
- Validate against schema

**Interfaces:**

```python
class NormalizationPipeline:
    def normalize_permit_type(self, raw_type: str) -> StandardPermitType:
        """Map to standard permit taxonomy"""
        
    def normalize_address(self, raw_address: str) -> CanonicalAddress:
        """Geocode and standardize address format"""
        
    def normalize_fee(self, raw_fee: str) -> FeeStructure:
        """Parse and structure fee information"""
        
    def normalize_timeline(self, raw_timeline: str) -> Duration:
        """Convert duration expressions to standard units"""
        
    def validate_schema(self, normalized_data: dict) -> ValidationResult:
        """Validate against knowledge graph schema"""
```

**Normalization Rules:**
- Permit types mapped to hierarchical taxonomy (Building → Residential → Addition)
- Addresses geocoded to lat/lon with jurisdiction boundaries
- Fees structured as base amount + calculation rules + conditions
- Timelines converted to business days with confidence intervals

### 3. Entity Extraction

**Responsibilities:**
- Identify permit-related entities in text
- Extract requirements, conditions, dependencies
- Parse zoning rules and restrictions
- Structure fee schedules and inspection requirements

**Interfaces:**

```python
class EntityExtractor:
    def extract_permit_types(self, text: str) -> List[PermitEntity]:
        """Identify permit types with context"""
        
    def extract_requirements(self, text: str) -> List[Requirement]:
        """Parse conditions, dependencies, exemptions"""
        
    def extract_zoning_rules(self, text: str) -> List[ZoningRule]:
        """Extract setbacks, height limits, use restrictions"""
        
    def extract_fees(self, text: str) -> List[FeeSchedule]:
        """Structure fee amounts and calculation methods"""
        
    def extract_inspections(self, text: str) -> List[InspectionRequirement]:
        """Parse inspection types and timing"""
```

**Entity Types:**
- PermitType: Category, subcategory, description, jurisdiction
- Requirement: Condition text, dependencies, exemptions, confidence
- ZoningRule: Rule type, measurements, applicability
- FeeSchedule: Base fee, calculation method, conditions
- InspectionRequirement: Type, timing, responsible party

### 4. Knowledge Graph Storage

**Responsibilities:**
- Store normalized permit rules as graph nodes
- Maintain relationships (dependencies, jurisdictions, versions)
- Support temporal queries for historical data
- Enable efficient graph traversal

**Schema:**

```
Nodes:
- City (name, state, jurisdiction_boundary)
- PermitType (category, subcategory, description)
- Requirement (text, conditions, confidence_score)
- Fee (amount, calculation_method, currency)
- Timeline (duration, unit, confidence_interval)
- ZoningRule (type, value, unit)
- Document (source_url, retrieval_date, content_hash)

Relationships:
- REQUIRES (PermitType → Requirement)
- DEPENDS_ON (PermitType → PermitType)
- APPLIES_IN (PermitType → City)
- HAS_FEE (PermitType → Fee)
- HAS_TIMELINE (PermitType → Timeline)
- GOVERNED_BY (PermitType → ZoningRule)
- SOURCED_FROM (PermitType → Document)
- VERSION_OF (PermitType → PermitType, with timestamp)
```

**Interfaces:**

```python
class KnowledgeGraphStore:
    def store_permit_rule(self, rule: PermitRule) -> NodeID:
        """Store normalized permit rule as graph node"""
        
    def create_relationship(self, from_node: NodeID, rel_type: str, to_node: NodeID, properties: dict):
        """Create typed relationship between nodes"""
        
    def query_by_address(self, address: CanonicalAddress) -> List[PermitRule]:
        """Retrieve all applicable rules for jurisdiction"""
        
    def query_by_permit_type(self, permit_type: StandardPermitType) -> PermitRule:
        """Retrieve complete permit rule with relationships"""
        
    def query_dependencies(self, permit_type: StandardPermitType) -> DependencyGraph:
        """Traverse dependency relationships"""
        
    def query_historical(self, permit_type: StandardPermitType, effective_date: datetime) -> PermitRule:
        """Retrieve rule version effective at specified date"""
```

### 5. Version Control System

**Responsibilities:**
- Detect changes in permit rules
- Create versioned snapshots with timestamps
- Maintain version lineage
- Support temporal queries

**Interfaces:**

```python
class VersionControlSystem:
    def detect_change(self, current: PermitRule, previous: PermitRule) -> ChangeSet:
        """Identify differences between versions"""
        
    def create_version(self, rule: PermitRule, change_desc: str) -> Version:
        """Create new version with timestamp"""
        
    def get_version_history(self, rule_id: str) -> List[Version]:
        """Retrieve all versions chronologically"""
        
    def get_effective_version(self, rule_id: str, date: datetime) -> Version:
        """Get version effective at specified date"""
        
    def resolve_conflict(self, versions: List[Version]) -> Version:
        """Handle conflicting versions from multiple sources"""
```

**Versioning Strategy:**
- Each change creates new version node linked via VERSION_OF relationship
- Versions include: timestamp, change description, source provenance, confidence
- Temporal queries use effective_date and expiration_date properties
- Conflicts flagged for manual resolution with source comparison

### 6. Query API

**Responsibilities:**
- Expose REST and GraphQL endpoints
- Handle authentication and rate limiting
- Return results with metadata and confidence scores
- Support complex queries with filtering

**REST Endpoints:**

```
GET /api/v1/permits/by-address?address={address}&date={date}
GET /api/v1/permits/by-type?type={permit_type}&jurisdiction={city}
GET /api/v1/permits/by-business?category={business_category}&address={address}
GET /api/v1/permits/{permit_id}/dependencies
GET /api/v1/permits/{permit_id}/timeline
GET /api/v1/permits/{permit_id}/history
GET /api/v1/jurisdictions/{city}/coverage
```

**GraphQL Schema:**

```graphql
type PermitRule {
  id: ID!
  permitType: PermitType!
  requirements: [Requirement!]!
  fees: [Fee!]!
  timeline: Timeline
  dependencies: [PermitRule!]!
  jurisdiction: City!
  effectiveDate: DateTime!
  expirationDate: DateTime
  source: Document!
  confidenceScore: Float!
}

type Query {
  permitsByAddress(address: String!, date: DateTime): [PermitRule!]!
  permitByType(type: String!, jurisdiction: String!): PermitRule
  permitsByBusiness(category: String!, address: String!): [PermitRule!]!
  permitDependencies(permitId: ID!): DependencyGraph!
  permitTimeline(permitId: ID!): TimelineEstimate!
}
```

**Response Format:**

```json
{
  "data": {
    "permitType": "Residential Addition",
    "requirements": [...],
    "fees": {...},
    "timeline": {...},
    "dependencies": [...]
  },
  "metadata": {
    "source": "https://oakland.gov/permits/...",
    "retrievalDate": "2025-12-01",
    "effectiveDate": "2025-01-01",
    "confidenceScore": 0.92,
    "version": "v2.3"
  }
}
```

### 7. Quality Monitoring

**Responsibilities:**
- Track data completeness by city and permit type
- Monitor ingestion health and parsing success rates
- Detect anomalies and data quality issues
- Generate alerts for administrators

**Interfaces:**

```python
class QualityMonitor:
    def calculate_completeness(self, city: str) -> CompletenessMetrics:
        """Calculate coverage by permit type and data field"""
        
    def track_parsing_success(self, ingestion_run: IngestionRun) -> SuccessMetrics:
        """Monitor parsing success rates and error types"""
        
    def detect_anomalies(self, data: PermitRule) -> List[Anomaly]:
        """Identify unusual values or patterns"""
        
    def generate_quality_report(self, city: str, date_range: tuple) -> QualityReport:
        """Comprehensive quality metrics for jurisdiction"""
```

**Metrics Tracked:**
- Completeness: % of permit types covered, % of required fields populated
- Accuracy: Confidence scores, validation pass rates
- Freshness: Days since last update, staleness alerts
- Parsing: Success rates by format, error types and frequencies
- Coverage: Cities covered, permit types per city, total rules

## Data Models

### Core Data Models

```python
@dataclass
class RawDocument:
    source_url: str
    content: bytes
    format: str  # 'pdf', 'html', 'xlsx', 'image'
    retrieval_date: datetime
    content_hash: str
    
@dataclass
class ExtractedText:
    text: str
    structure: dict  # Preserved headings, sections, tables
    metadata: dict
    confidence_score: float
    
@dataclass
class StandardPermitType:
    category: str  # 'Building', 'Electrical', 'Plumbing', etc.
    subcategory: str  # 'Residential', 'Commercial', etc.
    specific_type: str  # 'Addition', 'New Construction', etc.
    description: str
    
@dataclass
class CanonicalAddress:
    street_number: str
    street_name: str
    unit: Optional[str]
    city: str
    state: str
    zip_code: str
    latitude: float
    longitude: float
    jurisdiction: str
    
@dataclass
class FeeStructure:
    base_amount: Decimal
    currency: str
    calculation_method: str  # 'flat', 'per_sqft', 'percentage', etc.
    conditions: List[str]
    additional_fees: List[dict]
    
@dataclass
class Duration:
    value: int
    unit: str  # 'business_days', 'weeks', 'months'
    confidence_interval: tuple  # (min, max)
    
@dataclass
class Requirement:
    text: str
    conditions: List[str]
    dependencies: List[str]  # IDs of prerequisite permits
    exemptions: List[str]
    confidence_score: float
    
@dataclass
class PermitRule:
    id: str
    permit_type: StandardPermitType
    requirements: List[Requirement]
    fees: FeeStructure
    timeline: Duration
    jurisdiction: str
    effective_date: datetime
    expiration_date: Optional[datetime]
    source_document: str
    confidence_score: float
    version: str
    
@dataclass
class DependencyGraph:
    root_permit: str
    dependencies: List[dict]  # {permit_id, relationship_type, sequence}
    parallel_paths: List[List[str]]
    critical_path: List[str]
    total_timeline: Duration
    
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    completeness_score: float
```

### MVP Data Model (Oakland Focus)

For the Oakland MVP, we'll prioritize these permit types:
- Building Permits (Residential/Commercial)
- Electrical Permits
- Plumbing Permits
- Mechanical Permits
- ADU (Accessory Dwelling Unit) Permits
- Demolition Permits

Oakland-specific data sources:
- https://www.oaklandca.gov/services/building-permits
- Oakland Municipal Code (zoning)
- Planning and Building Department PDFs
- Fee schedules (typically PDF/HTML)


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Ingestion Properties

Property 1: PDF structure preservation
*For any* PDF document with known structure, extracting text should preserve the hierarchical structure of headings, sections, and tables
**Validates: Requirements 1.1**

Property 2: HTML semantic relationship preservation
*For any* HTML document with semantic markup, parsing and extraction should maintain the relationships between elements (parent-child, sibling, reference relationships)
**Validates: Requirements 1.2**

Property 3: Spreadsheet header mapping
*For any* spreadsheet with identifiable column headers, processing should correctly identify headers and map data rows to standardized fields
**Validates: Requirements 1.3**

Property 4: OCR confidence scoring
*For any* scanned document processed with OCR, the system should return extracted text with confidence scores for each text region
**Validates: Requirements 1.4**

Property 5: API endpoint preference
*For any* city source that provides both an API endpoint and scrapable content, the system should use the API endpoint and track any endpoint URL changes
**Validates: Requirements 1.5**

### Normalization Properties

Property 6: Permit type taxonomy mapping
*For any* permit type string extracted from source documents, normalization should map it to a valid entry in the standardized taxonomy
**Validates: Requirements 2.1**

Property 7: Address geocoding completeness
*For any* valid address string, normalization should produce a canonical address with all required fields (street, city, state, zip) and geocoded coordinates (latitude, longitude)
**Validates: Requirements 2.2**

Property 8: Fee structure extraction
*For any* fee information string, normalization should extract a structured fee with base amount, currency, and calculation method
**Validates: Requirements 2.3**

Property 9: Timeline standardization
*For any* duration expression (e.g., "2-3 weeks", "30 business days"), normalization should convert it to standardized time units with confidence intervals
**Validates: Requirements 2.4**

Property 10: Schema validation completeness
*For any* normalized data record, validation should check all required fields and flag any inconsistencies or missing data
**Validates: Requirements 2.5**

### Entity Extraction Properties

Property 11: Permit type identification
*For any* text containing permit descriptions, entity extraction should identify permit types and associate them with relevant metadata (jurisdiction, category, requirements)
**Validates: Requirements 3.1**

Property 12: Requirement parsing completeness
*For any* requirement description text, entity extraction should parse and structure conditions, dependencies, and exemptions as separate fields
**Validates: Requirements 3.2**

Property 13: Zoning rule measurement extraction
*For any* zoning rule text containing measurements, entity extraction should identify and extract setback distances, height limits, and use restrictions with units
**Validates: Requirements 3.3**

Property 14: Fee schedule structuring
*For any* fee schedule text, entity extraction should structure fee amounts, calculation methods, and applicable conditions into a queryable format
**Validates: Requirements 3.4**

Property 15: Inspection requirement decomposition
*For any* inspection requirement text, entity extraction should identify and separate inspection types, timing requirements, and responsible parties
**Validates: Requirements 3.5**

### Version Control Properties

Property 16: Change detection and versioning
*For any* permit rule that changes, the system should detect the change and create a new version with timestamp and change description while preserving the previous version
**Validates: Requirements 4.1**

Property 17: Version lineage preservation
*For any* permit rule with multiple versions, all previous versions should be preserved and linked via version relationships forming a complete lineage
**Validates: Requirements 4.2**

Property 18: Temporal query correctness
*For any* permit rule with versioned history and any query date, the system should return the version that was effective at that specific date
**Validates: Requirements 4.3**

Property 19: Conflict detection and blocking
*For any* set of conflicting versions from different sources, the system should flag the conflict and prevent publishing until resolution
**Validates: Requirements 4.4**

Property 20: Multi-source provenance tracking
*For any* permit rule sourced from multiple city sources, the system should track provenance for each source and assign confidence scores based on source authority
**Validates: Requirements 4.5**

### Query API Properties

Property 21: Address-based jurisdiction filtering
*For any* address query, the system should return all and only the permit rules applicable to that address's jurisdiction
**Validates: Requirements 5.1**

Property 22: Permit type query completeness
*For any* permit type query, the system should return all associated data including requirements, timelines, fees, and dependencies
**Validates: Requirements 5.2**

Property 23: Business category permit aggregation
*For any* business category and address, the system should return all required permits and licenses for that business type in that jurisdiction
**Validates: Requirements 5.3**

Property 24: Query result metadata completeness
*For any* query result, the response should include confidence scores and source citations for all returned permit rules
**Validates: Requirements 5.4**

Property 25: Ambiguous query handling
*For any* ambiguous query, the system should either return clarifying questions or multiple interpretations ranked by relevance
**Validates: Requirements 5.5**

### Timeline Estimation Properties

Property 26: Historical data-based estimation
*For any* permit request with available historical processing data, timeline estimates should be calculated using both historical data and current rules
**Validates: Requirements 6.1**

Property 27: Dependency timeline aggregation
*For any* permit with dependencies, the total timeline estimate should include the timelines of all dependent permits in the critical path
**Validates: Requirements 6.2**

Property 28: Confidence interval inclusion
*For any* timeline estimate, the response should include confidence intervals and factors that may affect the duration
**Validates: Requirements 6.3**

Property 29: Statistical model application
*For any* permit type with sufficient historical data, timeline estimates should differ from baseline estimates in a way that reflects the statistical model's predictions
**Validates: Requirements 6.4**

Property 30: Uncertainty indication for missing data
*For any* permit without timeline information, the system should explicitly indicate uncertainty and provide typical ranges from similar jurisdictions
**Validates: Requirements 6.5**

### Quality Monitoring Properties

Property 31: Quality metrics generation
*For any* completed ingestion run, the system should generate quality metrics including completeness scores, accuracy scores, and freshness scores
**Validates: Requirements 7.1**

Property 32: Parsing failure logging
*For any* parsing failure, the system should log an error with source document reference and trigger administrator notification
**Validates: Requirements 7.2**

Property 33: Anomaly flagging and tracking
*For any* detected data anomaly, the system should flag the record for manual review and track its resolution status
**Validates: Requirements 7.3**

Property 34: Source failure handling
*For any* city source endpoint that becomes unavailable, the system should alert administrators and attempt alternative retrieval methods
**Validates: Requirements 7.4**

Property 35: Coverage metrics tracking
*For any* update processing run, the system should track and update coverage metrics by city, permit type, and data field
**Validates: Requirements 7.5**

### Metadata and Transparency Properties

Property 36: Complete response metadata
*For any* API query result, the response should include all required metadata: source document references with URLs and retrieval dates, confidence scores for extracted fields, transformation indicators with original values, effective/expiration dates with version identifiers, and complete jurisdiction information (city, county, state)
**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**

### Dependency Analysis Properties

Property 37: Prerequisite identification
*For any* permit request, the system should identify all prerequisite permits and their requirements through dependency graph traversal
**Validates: Requirements 9.1**

Property 38: Dependency graph construction
*For any* permit with dependencies, the system should construct a dependency graph showing required sequences and relationships
**Validates: Requirements 9.2**

Property 39: Circular dependency detection
*For any* dependency graph with circular references, the system should detect the cycle, flag it as an error, and provide resolution guidance
**Validates: Requirements 9.3**

Property 40: Timing constraint inclusion
*For any* dependency relationship, the system should include timing constraints that specify required delays or sequences between dependent permits
**Validates: Requirements 9.4**

Property 41: Parallel path optimization
*For any* dependency graph with permits that can be processed simultaneously, the system should identify parallel paths and calculate optimized total timeline
**Validates: Requirements 9.5**

### City Administrator Properties

Property 42: Correction audit trails
*For any* correction submitted by a city administrator, the system should store the correction, update the knowledge graph, and maintain a complete audit trail
**Validates: Requirements 10.2**

Property 43: Update notifications
*For any* published update to jurisdiction data, the system should notify all subscribed city administrators for that jurisdiction
**Validates: Requirements 10.3**

Property 44: Correction-based learning
*For any* correction submitted by a city administrator, the system should update the knowledge graph and apply the correction pattern to improve future extractions of similar content
**Validates: Requirements 10.4**

Property 45: Official feed prioritization
*For any* city that provides both official structured data feeds and scrapable content, the system should prioritize the official feed and mark it as authoritative in all metadata
**Validates: Requirements 10.5**

## Error Handling

### Ingestion Errors

**Document Retrieval Failures:**
- Retry with exponential backoff (3 attempts)
- Log failure with source URL and error type
- Alert administrators if source consistently fails
- Attempt alternative sources if available

**Parsing Failures:**
- Log document hash and error details
- Flag for manual review
- Continue processing other documents
- Track parsing success rates by format

**OCR Low Confidence:**
- Flag documents with average confidence < 0.7
- Queue for manual verification
- Include confidence scores in all downstream processing
- Consider alternative OCR engines for flagged documents

### Normalization Errors

**Unmappable Permit Types:**
- Log original text and attempted mappings
- Use "Other" category with original text preserved
- Flag for taxonomy expansion
- Track frequency to prioritize taxonomy updates

**Geocoding Failures:**
- Attempt fuzzy matching with known addresses
- Use jurisdiction boundaries if exact geocoding fails
- Flag address for manual review
- Include geocoding confidence in metadata

**Invalid Fee Structures:**
- Preserve original text
- Flag for manual structuring
- Return partial structure if possible
- Track patterns for improved extraction

### Query Errors

**No Results Found:**
- Return empty result set with suggestions
- Log query for analysis
- Suggest similar jurisdictions or permit types
- Provide coverage information for jurisdiction

**Ambiguous Queries:**
- Return multiple interpretations with confidence scores
- Provide clarifying questions
- Log ambiguity patterns for improved query parsing

**Version Not Found:**
- Return nearest version with temporal distance
- Indicate that exact version is unavailable
- Suggest valid date ranges

### System Errors

**Database Connection Failures:**
- Retry with connection pool
- Fail over to read replica
- Return cached results if available
- Alert operations team

**Timeout Errors:**
- Return partial results with continuation token
- Log slow queries for optimization
- Implement query complexity limits

## Testing Strategy

### Unit Testing Approach

Unit tests will verify specific examples and edge cases for individual components:

**Ingestion Engine Tests:**
- Test PDF extraction with sample documents (simple, complex, scanned)
- Test HTML parsing with various DOM structures
- Test spreadsheet parsing with different header formats
- Test OCR with known low-quality scans
- Test API endpoint detection and usage

**Normalization Pipeline Tests:**
- Test permit type mapping with known examples
- Test address geocoding with valid and invalid addresses
- Test fee parsing with various formats ($100, $1.50/sqft, 2% of value)
- Test timeline parsing with different expressions
- Test schema validation with valid and invalid data

**Entity Extraction Tests:**
- Test requirement parsing with sample permit text
- Test zoning rule extraction with measurement variations
- Test fee schedule parsing with complex structures
- Test inspection requirement decomposition

**Version Control Tests:**
- Test version creation with sample changes
- Test temporal queries with known version history
- Test conflict detection with competing versions

**Query API Tests:**
- Test address queries with sample addresses
- Test permit type queries with known data
- Test error responses for invalid inputs

**Quality Monitoring Tests:**
- Test metrics calculation with sample ingestion runs
- Test anomaly detection with known anomalies
- Test alert generation for failures

### Property-Based Testing Approach

Property-based tests will verify universal properties across randomly generated inputs using a PBT library. For Python implementation, we will use **Hypothesis**. For TypeScript/JavaScript implementation, we will use **fast-check**.

**Configuration:**
- Each property test MUST run a minimum of 100 iterations
- Each property test MUST be tagged with a comment referencing the design document property
- Tag format: `# Feature: city-permitting-knowledge-graph, Property {number}: {property_text}`

**Property Test Categories:**

**Ingestion Properties (Properties 1-5):**
- Generate random PDFs with known structure, verify structure preservation
- Generate random HTML with semantic relationships, verify relationship preservation
- Generate random spreadsheets with headers, verify correct mapping
- Generate random images with text, verify OCR returns confidence scores
- Generate city sources with both API and scraping options, verify API preference

**Normalization Properties (Properties 6-10):**
- Generate random permit type strings, verify taxonomy mapping
- Generate random address strings, verify geocoding completeness
- Generate random fee strings, verify structured extraction
- Generate random duration expressions, verify standardization
- Generate random normalized data (valid and invalid), verify validation flags inconsistencies

**Entity Extraction Properties (Properties 11-15):**
- Generate random permit descriptions, verify entity identification
- Generate random requirement text, verify parsing completeness
- Generate random zoning text with measurements, verify extraction
- Generate random fee schedules, verify structuring
- Generate random inspection text, verify decomposition

**Version Control Properties (Properties 16-20):**
- Generate random permit rules and modifications, verify versioning
- Generate version chains, verify lineage preservation
- Generate versioned rules with dates, verify temporal query correctness
- Generate conflicting versions, verify conflict detection
- Generate multi-source rules, verify provenance tracking

**Query API Properties (Properties 21-25):**
- Generate random addresses with known rules, verify jurisdiction filtering
- Generate random permit types, verify query completeness
- Generate random business categories, verify permit aggregation
- Generate random queries, verify metadata completeness
- Generate ambiguous queries, verify handling

**Timeline Properties (Properties 26-30):**
- Generate permits with historical data, verify estimation uses data
- Generate permits with dependencies, verify timeline aggregation
- Generate timeline requests, verify confidence intervals
- Generate permits with/without historical data, verify statistical model application
- Generate permits without timeline data, verify uncertainty indication

**Quality Monitoring Properties (Properties 31-35):**
- Generate ingestion runs, verify metrics generation
- Trigger parsing failures, verify logging and notifications
- Inject anomalous data, verify flagging and tracking
- Simulate source failures, verify alerts and alternatives
- Process updates, verify coverage tracking

**Metadata Properties (Property 36):**
- Generate random queries, verify all metadata fields present

**Dependency Properties (Properties 37-41):**
- Generate permits with prerequisites, verify identification
- Generate dependency chains, verify graph construction
- Generate circular dependencies, verify detection
- Generate dependencies, verify timing constraints
- Generate parallel dependencies, verify optimization

**Administrator Properties (Properties 42-45):**
- Generate corrections, verify audit trails
- Publish updates, verify notifications
- Submit corrections, verify learning
- Provide official feeds, verify prioritization

### Integration Testing

Integration tests will verify end-to-end workflows:

**Oakland MVP Integration Tests:**
- Ingest sample Oakland permit documents
- Verify complete pipeline: ingestion → normalization → entity extraction → storage
- Query by Oakland addresses and verify results
- Test version control with Oakland rule changes
- Verify quality metrics for Oakland data

**Multi-City Integration Tests:**
- Ingest documents from Oakland, SF, and Berkeley
- Verify jurisdiction isolation
- Test cross-city queries
- Verify coverage metrics across cities

**API Integration Tests:**
- Test all REST endpoints with realistic queries
- Test GraphQL queries with complex selections
- Verify rate limiting and authentication
- Test error responses and edge cases

### Performance Testing

**Load Testing:**
- Test query throughput (target: 1000 queries/second)
- Test concurrent ingestion jobs
- Test large document processing (100+ page PDFs)

**Scalability Testing:**
- Test with 10, 100, 1000 cities
- Test with 10K, 100K, 1M permit rules
- Verify query performance remains acceptable

### Test Data Strategy

**Synthetic Data Generation:**
- Create generators for each entity type
- Use realistic distributions (permit types, fees, timelines)
- Include edge cases (empty fields, extreme values, special characters)

**Real Data Samples:**
- Collect sample documents from Oakland (with permission)
- Anonymize if necessary
- Use for integration and validation testing

**Continuous Testing:**
- Run unit tests on every commit
- Run property tests nightly (due to longer execution time)
- Run integration tests on staging deployments
- Monitor production with synthetic queries

## Implementation Phases

### Phase 1: Oakland MVP (Weeks 1-8)

**Goals:**
- Ingest Oakland building permit data
- Normalize to standard schema
- Basic entity extraction
- Simple query API
- Version control foundation

**Deliverables:**
- Working ingestion for Oakland PDFs and HTML
- Normalized permit rules in knowledge graph
- REST API with address and permit type queries
- Basic version tracking
- Unit tests and initial property tests

**Success Criteria:**
- Successfully ingest 80%+ of Oakland permit types
- Query API returns accurate results for test addresses
- All core property tests passing

### Phase 2: SF and Berkeley Expansion (Weeks 9-12)

**Goals:**
- Extend ingestion to SF and Berkeley
- Improve normalization for format variations
- Enhanced entity extraction
- Dependency analysis
- Timeline estimation

**Deliverables:**
- Multi-city ingestion pipeline
- Improved normalization handling format differences
- Dependency graph construction
- Timeline estimation with confidence intervals
- Expanded test coverage

**Success Criteria:**
- Successfully ingest 70%+ of permit types from all three cities
- Dependency analysis working for common permit chains
- Timeline estimates within 20% of actual processing times

### Phase 3: Seattle Area Expansion (Weeks 13-16)

**Goals:**
- Scale to Seattle, Bellevue, Tacoma
- Quality monitoring dashboard
- GraphQL API
- City administrator tools
- Production readiness

**Deliverables:**
- Seattle area city ingestion
- Quality monitoring and alerting
- GraphQL API with complex queries
- City validation interface
- Production deployment

**Success Criteria:**
- 6+ cities with 70%+ coverage
- Quality monitoring catching issues before API consumers
- GraphQL API supporting all query patterns
- Production uptime > 99.5%

### Phase 4: API Productization (Weeks 17-20)

**Goals:**
- API documentation and developer portal
- Rate limiting and authentication
- Usage analytics
- Customer onboarding tools
- Commercial readiness

**Deliverables:**
- Comprehensive API documentation
- Developer portal with examples
- Tiered access with rate limiting
- Usage dashboards
- Customer success tools

**Success Criteria:**
- API documentation complete with examples
- First pilot customers onboarded
- Usage analytics tracking adoption
- Revenue-generating API access tiers

## Future Enhancements

### Advanced Features

**Machine Learning Improvements:**
- Fine-tune NER models on permitting domain
- Improve OCR with domain-specific training
- Predict permit approval likelihood
- Anomaly detection for unusual requirements

**Expanded Coverage:**
- Scale to top 100 US cities
- International expansion (Canada, UK, Australia)
- County and state-level permits
- Federal permits and regulations

**Enhanced Intelligence:**
- Natural language query interface
- Permit application assistance
- Compliance checking for proposed projects
- Regulatory change impact analysis

**Integration Ecosystem:**
- Zillow/Redfin property data integration
- Contractor platform integrations
- City permitting system APIs
- GIS and mapping integrations

### Scalability Improvements

**Infrastructure:**
- Distributed graph database
- CDN for frequently accessed data
- Regional deployments for low latency
- Automated scaling based on load

**Data Pipeline:**
- Parallel ingestion across cities
- Incremental updates instead of full re-ingestion
- Real-time change detection
- Automated quality improvement

## Risks and Mitigations

### Technical Risks

**Risk: OCR accuracy insufficient for scanned documents**
- Mitigation: Use multiple OCR engines, manual review for low confidence, prioritize cities with digital documents

**Risk: Permit type taxonomy too rigid or too flexible**
- Mitigation: Iterative taxonomy development, allow custom categories, track unmappable types

**Risk: Graph database performance degrades with scale**
- Mitigation: Benchmark early, optimize queries, consider sharding strategies, use caching aggressively

### Data Risks

**Risk: City sources change frequently breaking ingestion**
- Mitigation: Monitor sources daily, maintain multiple retrieval strategies, build relationships with city IT departments

**Risk: Data quality varies significantly across cities**
- Mitigation: Quality scoring per city, transparent confidence metrics, manual review for critical data

**Risk: Legal/copyright issues with city data**
- Mitigation: Verify public domain status, attribute sources properly, establish data sharing agreements

### Business Risks

**Risk: Cities object to data scraping**
- Mitigation: Offer validation tools to cities, provide value back to municipalities, establish partnerships

**Risk: Competitors emerge with similar offerings**
- Mitigation: Focus on data quality and coverage, build network effects, establish customer relationships early

**Risk: Market adoption slower than expected**
- Mitigation: Start with pilot customers, prove ROI clearly, offer free tier for adoption, focus on high-value use cases
