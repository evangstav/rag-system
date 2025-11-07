# RAG System Documentation

**Welcome to the RAG System documentation!** This guide will help you navigate all available documentation.

---

## 🚀 Quick Start

**New to the project?** Start here:

1. [**Architecture Overview**](./architecture/ARCHITECTURE.md) - Understand the system design
2. [**Quick Start Guide**](./guides/QUICK_START_GUIDE.md) - Get running in 2 hours
3. [**Setup Guide**](./guides/SETUP_GUIDE.md) - Detailed setup instructions

---

## 📚 Documentation Index

### Architecture & Design

**Start here to understand how the system works:**

- [**System Architecture**](./architecture/ARCHITECTURE.md)
  - Complete architecture overview with diagrams
  - Technology stack and design decisions
  - Database schema and data flow
  - Security and deployment considerations

- [**Backend Services**](./architecture/BACKEND_SERVICES.md)
  - Detailed RAG components documentation
  - All 13+ specialized services explained
  - Usage examples and code snippets
  - Configuration and troubleshooting

---

### Getting Started

**Setup and configuration guides:**

- [**Quick Start Guide**](./guides/QUICK_START_GUIDE.md)
  - Get running in 2 hours
  - Step-by-step MVP setup
  - Essential configuration

- [**Setup Guide**](./guides/SETUP_GUIDE.md)
  - Complete installation guide
  - Docker services setup
  - Database migrations
  - Environment configuration

---

### Development

**For contributors and developers:**

- [**Complete Development Guide**](./development/COMPLETE_DEVELOPMENT_GUIDE.md)
  - Comprehensive development reference
  - Code patterns and conventions
  - API design guidelines
  - Best practices

- [**E2E Testing**](./development/E2E_TESTING.md)
  - Automated RAG evaluation testing
  - Testing framework usage
  - Creating test suites

---

### Testing & Evaluation

**RAG evaluation and quality assurance:**

- [**Evaluation Suite Guide**](./evaluation/EVALUATION_SUITE_GUIDE.md)
  - Creating test cases and suites
  - Standard IR metrics (Precision, Recall, NDCG, MAP)
  - Comparing implementations
  - Performance tracking

- [**Testing Workflow**](./testing/TESTING_WORKFLOW.md)
  - Testing best practices
  - Integration testing
  - E2E testing setup

---

### Plans & Proposals

**Implementation plans and future features:**

- [**Plans Directory**](./plans/README.md)
  - Active implementation plans
  - Completed features (archived)
  - Research and proposals

**Active Plans:**
- Query Rewriting Enhancement (proposed)
- RAG Retrieval Improvements (reference)

**Completed Plans:**
- ✅ Hybrid Search (deployed)
- ✅ Reranking & Deduplication (deployed)
- ✅ PostgreSQL BM25 (deployed)

---

### Reference

**Technical references and analysis:**

- [**RAG Improvements Analysis**](./reference/RAG_IMPROVEMENTS.md)
  - Historical analysis of RAG improvements
  - What's been implemented
  - Remaining opportunities
  - Performance considerations

- [**Code Snippets Collection**](./reference/RAG_SYSTEM_SNIPPET_COLLECTION.md)
  - Common patterns and examples
  - Reusable code snippets
  - Implementation templates

---

### Project Status

**Current state and roadmap:**

- [**Project Status**](./status/PROJECT_STATUS.md)
  - Implemented features
  - Known limitations
  - Roadmap and next steps
  - Quick wins and priorities

---

## 🎯 Documentation by Role

### I'm a New Developer

1. Read [Architecture Overview](./architecture/ARCHITECTURE.md)
2. Follow [Quick Start Guide](./guides/QUICK_START_GUIDE.md)
3. Review [Backend Services](./architecture/BACKEND_SERVICES.md)
4. Check [Project Status](./status/PROJECT_STATUS.md) for current state
5. See [Complete Development Guide](./development/COMPLETE_DEVELOPMENT_GUIDE.md) for patterns

### I'm Implementing a Feature

1. Check [Plans Directory](./plans/README.md) for existing proposals
2. Review [Backend Services](./architecture/BACKEND_SERVICES.md) for component APIs
3. See [Code Snippets](./reference/RAG_SYSTEM_SNIPPET_COLLECTION.md) for examples
4. Follow patterns in [Complete Development Guide](./development/COMPLETE_DEVELOPMENT_GUIDE.md)

### I'm Testing the System

1. Read [Evaluation Suite Guide](./evaluation/EVALUATION_SUITE_GUIDE.md)
2. Follow [E2E Testing](./development/E2E_TESTING.md) workflow
3. Use [Testing Workflow](./testing/TESTING_WORKFLOW.md) best practices
4. Check [Project Status](./status/PROJECT_STATUS.md) for known issues

### I'm Deploying to Production

1. Review [Architecture Overview](./architecture/ARCHITECTURE.md) deployment section
2. Follow [Setup Guide](./guides/SETUP_GUIDE.md) for production config
3. Check [Backend Services](./architecture/BACKEND_SERVICES.md) for configuration
4. Review [Project Status](./status/PROJECT_STATUS.md) for production readiness

---

## 📖 Documentation Standards

### File Organization

```
docs/
├── README.md                    # This file (documentation index)
├── architecture/                # System design and architecture
├── development/                 # Development guides and workflows
├── evaluation/                  # Testing and evaluation frameworks
├── guides/                      # Getting started and setup guides
├── plans/                       # Implementation plans and proposals
│   └── archive/                # Completed implementation plans
├── reference/                   # Technical references and snippets
├── status/                      # Project status and roadmap
└── testing/                     # Testing workflows and best practices
```

### Documentation Types

- **Architecture Docs**: High-level system design and component relationships
- **Guides**: Step-by-step instructions for specific tasks
- **Reference**: Technical details, APIs, and code examples
- **Plans**: Proposals and implementation roadmaps
- **Status**: Current state, features, and roadmap

---

## 🔄 Keeping Documentation Updated

### When to Update

- **Architecture changes**: Update `architecture/` docs
- **New features**: Update `status/PROJECT_STATUS.md` and relevant guides
- **API changes**: Update `architecture/BACKEND_SERVICES.md`
- **Deployment changes**: Update `guides/SETUP_GUIDE.md`
- **New proposals**: Add to `plans/` directory

### Documentation Workflow

1. **Plan**: Create proposal in `plans/`
2. **Implement**: Build the feature
3. **Document**: Update architecture and reference docs
4. **Archive**: Move completed plans to `plans/archive/`
5. **Update Status**: Mark as completed in `status/PROJECT_STATUS.md`

---

## 🤝 Contributing to Documentation

### Best Practices

- **Be Concise**: Get to the point quickly
- **Use Examples**: Show code snippets and diagrams
- **Link Related Docs**: Help readers find related information
- **Keep Updated**: Update docs alongside code changes
- **Test Instructions**: Verify setup steps actually work

### Markdown Standards

- Use clear headings (`##`, `###`)
- Include code blocks with language tags
- Add diagrams for complex concepts
- Use tables for comparisons
- Include cross-references with relative links

---

## 📞 Getting Help

### Documentation Issues

- Found outdated info? Open an issue
- Missing documentation? Submit a PR
- Unclear instructions? Ask for clarification

### Other Resources

- [**CLAUDE.md**](../CLAUDE.md) - Project overview and conventions
- [**README.md**](../README.md) - Repository README
- [**Project Status**](./status/PROJECT_STATUS.md) - Current priorities

---

## 🗺️ Documentation Roadmap

### Current State (November 2024)

✅ Comprehensive architecture documentation
✅ Backend services fully documented
✅ Setup and quick start guides
✅ RAG evaluation framework documented
✅ Plans and status tracking
✅ Historical analysis preserved

### Future Improvements

- [ ] API reference auto-generation (OpenAPI)
- [ ] Video tutorials for common tasks
- [ ] Interactive architecture diagrams
- [ ] Performance tuning guides
- [ ] Troubleshooting playbooks
- [ ] Contribution guidelines

---

**Last Updated**: November 7, 2024
**Documentation Status**: ✅ Complete and Current
