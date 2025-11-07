# Implementation Plans

This directory contains detailed implementation plans and proposals for major features and improvements to the RAG system.

## Purpose

Implementation plans serve as:
- **Roadmaps** - Step-by-step guides for implementing complex features
- **Research Documentation** - Background research and industry best practices
- **Technical Specifications** - Detailed architecture and code examples
- **Project Memory** - Historical context for future maintainers

---

## Active Plans

### 1. Query Rewriting Enhancement

**Status**: 📋 Ready for Implementation
**Priority**: ⭐ High
**Expected Impact**: +25-35% retrieval quality improvement

A comprehensive plan to implement advanced query rewriting strategies for improved RAG retrieval performance.

**Documents**:
- [**Query Rewriting Improvement Proposal**](./QUERY_REWRITING_IMPROVEMENT_PROPOSAL.md) - Complete implementation plan with research-backed strategies, architecture design, and code examples
- [**Comparative Analysis**](./QUERY_REWRITING_VS_OTHER_IMPROVEMENTS.md) - Analysis of how query rewriting fits with other RAG improvements (hybrid search, reranking)

**Key Strategies**:
1. **Multi-Query Generation (RAG-Fusion)** - Generate 3-5 diverse query variants (+25-35% recall)
2. **HyDE (Hypothetical Document Embeddings)** - Generate hypothetical answers (+15-20% for factual queries)
3. **Step-Back Prompting** - Create abstract queries for conceptual retrieval (+30% on complex queries)
4. **Query Decomposition** - Break complex queries into sub-queries (+40% on multi-part questions)
5. **Adaptive Strategy Selection** - Automatically choose the best strategy per query type

**Timeline**: 4 weeks
- Week 1: Foundation (protocols, fusion, infrastructure)
- Week 2: Core strategies (Multi-Query, HyDE, Step-Back)
- Week 3: Advanced features (Decomposition, Adaptive selection)
- Week 4: Optimization and tuning

**Research References**:
- HyDE: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022)
- RAG-Fusion: "Forget RAG, the Future is RAG-Fusion" (Rackauckas, 2023)
- Step-Back Prompting: "Take a Step Back: Evoking Reasoning via Abstraction" (Zheng et al., 2023)
- Query Rewriting for RAG: "Query Rewriting for Retrieval-Augmented Large Language Models" (Ma et al., 2023)

---

### 2. Retrieval Improvements (Reranking & Deduplication)

**Status**: ✅ Deployed
**Implementation Date**: October 2024
**Measured Impact**: Successfully deployed with cross-encoder reranking and MMR deduplication

Comprehensive proposal and implementation for improving RAG retrieval post-processing through reranking and deduplication.

**Documents**:
- [**RAG Retrieval Improvement Proposal**](./RAG_RETRIEVAL_IMPROVEMENT_PROPOSAL.md) - Multi-stage retrieval pipeline with complete implementation code
- [**Reranking & Deduplication Implementation**](./archive/RERANKING_DEDUPLICATION_IMPLEMENTATION.md) - Completed implementation plan (archived)

**Implemented Features**:
- ✅ **Cross-encoder reranking** using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- ✅ **MMR deduplication** for result diversification
- ✅ **Token-based deduplication** as alternative strategy
- ✅ **Feature flags** for configuration (`ENABLE_RERANKING`, configurable via settings)
- ✅ **Multiple reranker options** supported via protocol pattern

**Current Implementation**:
- `backend/app/services/rag/reranker.py` - Cross-encoder reranking
- `backend/app/services/rag/deduplication.py` - MMR and token-based deduplication
- Integrated into `RAGService` with optional enable/disable

---

## Completed Plans (Archived)

### 3. Hybrid Search (Semantic + BM25)

**Status**: ✅ Deployed
**Implementation Date**: October 2024

Successfully implemented hybrid search combining semantic (Qdrant) and keyword (PostgreSQL BM25) search with Reciprocal Rank Fusion.

**Documents**:
- [**PostgreSQL BM25 Migration**](./archive/POSTGRES_BM25_MIGRATION.md) - Complete migration from in-memory to PostgreSQL-backed BM25

**Implemented Features**:
- ✅ **PostgreSQL Full-Text Search** with GIN indexes for persistent BM25
- ✅ **Hybrid search service** combining semantic + keyword search
- ✅ **Reciprocal Rank Fusion (RRF)** for result merging
- ✅ **Configurable weights** for semantic vs keyword balance
- ✅ **Automatic tsvector maintenance** via PostgreSQL triggers

**Current Implementation**:
- `backend/app/services/rag/postgres_bm25.py` - PostgreSQL BM25 index
- `backend/app/services/rag/hybrid_search.py` - Hybrid search orchestration
- `backend/app/models/database.py` - BM25Document model

---

## Plan Structure

Each implementation plan should include:

1. **Executive Summary** - Overview of the proposal and expected impact
2. **Current State Analysis** - What exists today and what's missing
3. **Proposed Solution** - Detailed technical approach with research backing
4. **Architecture & Design** - Code structure, protocols, and implementation examples
5. **Implementation Roadmap** - Phased plan with timelines and deliverables
6. **Evaluation Plan** - Metrics, benchmarks, and success criteria
7. **Cost-Benefit Analysis** - Trade-offs and ROI considerations
8. **References** - Research papers and industry best practices

---

## Implementation Workflow

```
Research → Plan → Review → Implement → Test → Deploy → Monitor
   ↓         ↓       ↓         ↓         ↓       ↓        ↓
  Docs   This Dir  PR Review  Local   Staging  Prod   Analytics
```

1. **Research phase** → Create proposal document here
2. **Planning phase** → Create implementation plan here
3. **Review phase** → Submit PR with plans for team review
4. **Implementation** → Execute plan locally or via PR
5. **Testing** → Run evaluation suite (see [evaluation docs](../evaluation/))
6. **Deployment** → Gradual rollout with feature flags
7. **Monitoring** → Track metrics and adjust as needed

---

## Status Tracking

Plans can have the following statuses:

- **Research** - Gathering information and best practices
- **Proposed** - Plan written, awaiting review
- **Approved** - Reviewed and ready to implement
- **In Progress** - Currently being implemented
- **Testing** - Implementation complete, under evaluation
- **Deployed** - Live in production
- **Archived** - Superseded or no longer relevant

---

## Related Documentation

- [Development Guides](../development/) - Implementation guides and development workflows
- [Evaluation Suite](../evaluation/) - Testing and metrics framework
- [Reference Docs](../reference/) - Code snippets and architectural patterns
- [Project Status](../status/) - Current state and next steps

---

## Contributing Plans

When proposing a new major feature or improvement:

1. Research current best practices and industry standards
2. Document the current state and identify gaps
3. Propose a solution with clear technical architecture
4. Include implementation timeline and success metrics
5. Create a plan document in this directory
6. Update this README with your plan summary

**Note**: Plans should be research-backed, actionable, and include enough detail for implementation without being overly prescriptive about implementation details.

---

## Questions?

- Check [CLAUDE.md](../../CLAUDE.md) for project overview
- See [PROJECT_STATUS.md](../status/PROJECT_STATUS.md) for current priorities
- Review [COMPLETE_DEVELOPMENT_GUIDE.md](../development/COMPLETE_DEVELOPMENT_GUIDE.md) for architecture details
