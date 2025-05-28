# Subplan 11: Security

## Description
Security requirements for the system.

## Steps
- All sensitive data encrypted with AWS KMS
- No user-identifying information stored
- Ensure proper IAM permissions for KMS and DynamoDB
- Use least-privilege IAM roles for Lambda/API
- Rotate KMS keys periodically
- Audit DynamoDB and KMS access logs

## Pseudocode
# (IAM and KMS permissions are managed via AWS policies, not in code) 