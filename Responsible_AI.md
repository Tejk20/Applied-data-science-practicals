# Responsible AI Checklist & Governance Report

## 1. Fairness & Demographic Neutrality
- Personal identifiers (`phone_no_`, `roll_no_`) are dropped prior to model training.
- Predictions depend strictly on academic evaluation metrics (`math`, `physics`, `chemistry`).

## 2. Privacy & Data Protection
- Raw student inputs are processed in-memory and never logged to external non-volatile storage.

## 3. Explainability & Auditability
- Integrated SHAP tree explainability allows educators and students to audit individual feature contributions.

## 4. Model Monitoring & Continuous Drift Verification
- Out-of-sample accuracy audits (`X_test`) prevent reliance on overfitted training metrics.
