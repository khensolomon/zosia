# NMT Data Management Platform – Project Specification (Updated)

## Overview

The platform is designed to manage, collect, and collaborate on raw data for Neural Machine Translation (NMT), including words, sentences, parallel text, and corpus content. The aim is to support structured data entry, user collaboration, training data preparation, and live translation testing.

---

## Frontend Architecture

- The site may adopt a **Single Page Application (SPA)** architecture.
- The frontend must:
  - Be responsive across modern devices.
  - Use clean, semantic HTML and modular CSS.
  - Avoid decorative images – all visuals via **pure CSS**.
  - Support **light**, **dark**, and **system (auto)** themes.
  - Provide **easy language switching** (default: English).
  - Be accessible and follow **WCAG** guidelines.
  - Avoid use of **fancy emojis** anywhere in UI or text.

---

## Backend Requirements

- Built with a **popular and easy Python web framework** (to be selected).
- Provide a secure **REST API** for SPA communication.
- Must be capable of:
  - Validating and filtering requests.
  - Exporting structured training datasets, for example:  
    - Get all corpus data, merge and remove duplicates, break sentences into lines.  
    - Extract words and definitions or translations as requested.  
    - Support flexible export queries for model training data.
  - Handling large, unformatted input files using flat-file storage in a designated directory.
- Should enable future integration of model training or testing features.
- Support exporting required data for training models, customizable per admin or collaborator requests.

---

## Core Features

### Data Forms & Management

#### 1. Word Form

- Required: Source Word
- Optional (but encouraged):  
  - Grammar Type  
  - Meaning / Definition  
  - Example / Usage  
  - Synonym / Antonym  
  - Word Origin  
  - English Translation

#### 2. Sentence / Phrase Form

- Source Sentence
- Target Sentence (translated)

#### 3. Corpus Form

- Open format: Upload or input of parallel corpus or document-based text.

---

## User Management

- Use **Google Sign-In** (OAuth) to simplify authentication and avoid local password handling.
- **Collaborator Panel**:
  - Users can add/update/remove their own entries.
  - View contribution metrics (words added, translations submitted, etc.).
  - Display financial or point-based rewards if such a system is implemented.
- **Admin Panel**:
  - Manage users and raw data.
  - Monitor collaborator contributions and analytics.
  - Prepare data exports for training purposes.
  - Display financial or reward-related data associated with collaborators.

---

## UI Language Support

- Default UI language: **English**.
- Must support easy **externalization** of UI text (no hardcoded strings).
- Language switching should be possible via config or user settings.

---

## Design Guidelines

- **Timeless, elegant, minimal** aesthetic.
- Fully **responsive** layout.
- Limited, **neutral color palette**.
- **No emoji** or trendy visuals.
- Pure CSS for all design — no image-based decorations.
- Must support keyboard navigation and semantic HTML for accessibility.
- Aim for performance-conscious design (modular CSS, fast rendering, minimal layout shifts).

---

## Database

- **MySQL** is used as the main relational database.
- For large or unformatted raw data, **flat file storage** in a specified directory will be used instead of storing blobs in MySQL.

---

## Notes

- SPA (if used) must gracefully communicate with the backend via a **secure, filtered API**.
- Export functions (e.g., for training datasets) should be designed early, not bolted on later.
- Future integration with model testing (live translation demo) is a key feature.
- Avoid fancy emojis anywhere in the platform.
- Backend must support export features tailored to collaborator/admin needs.
- Collaborator rewards (financial or points) are a possible future feature with appropriate UI support.
