These worked examples demonstrate the expected extraction behavior across the
document layouts you will encounter. Each shows an input excerpt and the
`extract_document` tool input it should produce. Study how `null` is used for
absent fields and how confidence tracks evidence strength.

---

## Example A — Research paper with inline citations

**Input excerpt**

> Deep Contrastive Learning for Low-Resource Translation
> Smith et al. (2023). Journal of Machine Learning, vol. 24.
> Contact: a.smith@univ.edu  •  DOI: 10.1145/3580305.3599881
> As shown by Smith et al. (2023) and Lee (2021), contrastive objectives...

**Extraction**

```json
{
  "document_id": {"value": "example_a", "confidence": 1.0},
  "title": {"value": "Deep Contrastive Learning for Low-Resource Translation", "confidence": 0.97},
  "document_type": {"value": "research_paper", "confidence": 0.96},
  "summary": {"value": "A machine-learning paper introducing contrastive learning objectives for low-resource translation.", "confidence": 0.9},
  "other_document_type_detail": null,
  "publication_date": {"value": "2023-01-01", "confidence": 0.55},
  "author": {"value": "Smith et al.", "confidence": 0.92},
  "organization": {"value": "Journal of Machine Learning", "confidence": 0.7},
  "email": {"value": "a.smith@univ.edu", "confidence": 0.98},
  "phone_number": {"value": null, "confidence": 0.99},
  "doi": {"value": "10.1145/3580305.3599881", "confidence": 0.98},
  "citation_count": {"value": null, "confidence": 0.95},
  "overall_confidence": 0.88
}
```

Note: only a year was given, so `publication_date` is a low-confidence
`2023-01-01`; no phone or citation count appears, so both are `null`.

---

## Example B — Research paper with a bibliography

**Input excerpt**

> Title: Sparse Attention at Scale
> Authors: J. Okafor, R. Mehta
> References:
>   [1] Vaswani et al. Attention Is All You Need. 2017.
>   [2] Okafor, J. Efficient Transformers. 2022.
> Cited by: 412

**Extraction**

```json
{
  "document_id": {"value": "example_b", "confidence": 1.0},
  "title": {"value": "Sparse Attention at Scale", "confidence": 0.98},
  "document_type": {"value": "research_paper", "confidence": 0.95},
  "summary": {"value": "A research paper on sparse attention mechanisms for large-scale transformer models.", "confidence": 0.88},
  "other_document_type_detail": null,
  "publication_date": {"value": null, "confidence": 0.9},
  "author": {"value": "J. Okafor, R. Mehta", "confidence": 0.95},
  "organization": {"value": null, "confidence": 0.85},
  "email": {"value": null, "confidence": 0.97},
  "phone_number": {"value": null, "confidence": 0.99},
  "doi": {"value": null, "confidence": 0.9},
  "citation_count": {"value": 412, "confidence": 0.93},
  "overall_confidence": 0.87
}
```

Note: `citation_count` is a real integer (`412`) parsed from "Cited by: 412";
no DOI or email is present, so both are `null`.

---

## Example C — Narrative report

**Input excerpt**

> Q3 Field Operations Report
> Prepared by the Logistics Division, Northwind Inc.
> Over the third quarter, regional throughput rose 12% while incident rates...

**Extraction**

```json
{
  "document_id": {"value": "example_c", "confidence": 1.0},
  "title": {"value": "Q3 Field Operations Report", "confidence": 0.95},
  "document_type": {"value": "report", "confidence": 0.94},
  "summary": {"value": "A quarterly field-operations report describing throughput gains and incident-rate changes for Q3.", "confidence": 0.9},
  "other_document_type_detail": null,
  "publication_date": {"value": null, "confidence": 0.8},
  "author": {"value": "Logistics Division", "confidence": 0.8},
  "organization": {"value": "Northwind Inc.", "confidence": 0.93},
  "email": {"value": null, "confidence": 0.98},
  "phone_number": {"value": null, "confidence": 0.98},
  "doi": {"value": null, "confidence": 0.99},
  "citation_count": {"value": null, "confidence": 0.99},
  "overall_confidence": 0.9
}
```

---

## Example D — Structured table format

**Input excerpt**

> | Field        | Value                    |
> | ------------ | ------------------------ |
> | Title        | Vendor Services Agreement|
> | Author       | John Doe                 |
> | Organization | Acme Corp                |
> | Email        | legal@acme.com           |

**Extraction**

```json
{
  "document_id": {"value": "example_d", "confidence": 1.0},
  "title": {"value": "Vendor Services Agreement", "confidence": 0.97},
  "document_type": {"value": "contract", "confidence": 0.9},
  "summary": {"value": "A vendor services agreement between the issuing organization and a counterparty.", "confidence": 0.82},
  "other_document_type_detail": null,
  "publication_date": {"value": null, "confidence": 0.9},
  "author": {"value": "John Doe", "confidence": 0.97},
  "organization": {"value": "Acme Corp", "confidence": 0.96},
  "email": {"value": "legal@acme.com", "confidence": 0.97},
  "phone_number": {"value": null, "confidence": 0.98},
  "doi": {"value": null, "confidence": 0.99},
  "citation_count": {"value": null, "confidence": 0.99},
  "overall_confidence": 0.92
}
```

Note: values come straight from the table cells, so field confidences are high
even though the prose is minimal.
