# Public archive publication checklist

Recommended GitHub repository name:

`DEx-FCL-JNSM-Reproducibility`

Recommended description:

`Executable DEx-FCL implementation, deterministic benchmark protocol, real CICIoT2023/Edge-IIoTset results, tables and figures for the JNSM manuscript.`

## GitHub

1. Create a **Public** repository named `DEx-FCL-JNSM-Reproducibility`.
2. Do not initialize it with a README, because this bundle already contains one.
3. Upload the contents of this repository directory, preserving folders.
4. Commit with message: `Release reproducibility package v1.0.0`.
5. Create GitHub release/tag `v1.0.0`.

## Zenodo DOI via GitHub integration

Current Zenodo documentation states that, after connecting GitHub, a repository can be enabled from the Zenodo GitHub page and subsequent GitHub releases are ingested and archived. Once the repository is enabled, create the GitHub `v1.0.0` release and wait for Zenodo to process it. The Zenodo record then provides a version DOI and a concept DOI.

Suggested metadata:

- **Title:** DEx-FCL: Reproducibility package for adaptive Edge-IoT security management
- **Resource type:** Software
- **Version:** 1.0.0
- **Creator:** Arul Selvam P
- **Publication date:** 2026-08-29
- **Keywords:** federated learning; continual learning; intrusion detection; Edge-IoT; concept drift; zero-day detection; explainable AI; cybersecurity
- **Related manuscript title:** DEx-FCL: Drift- and Explanation-Consistent Federated Continual Learning for Adaptive Edge-IoT Security Management

After Zenodo assigns the DOI, use the version DOI in the manuscript Code Availability statement and optionally add the GitHub repository URL as a second link.
