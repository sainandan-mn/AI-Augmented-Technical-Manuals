# AI-Augmented Technical Manuals - Streamlit Gemini RAG POC

This proof of concept uses the supplied A320 ATA 32-42 brake-system manual extracts:

- AMM: maintenance task and safety procedure
- IPC: parts and effectivity matrix
- WDM: connector, wiring, and pin-out data
- FIM: fault-isolation procedure

The Streamlit UI starts with the requested aircraft configuration panel:

```text
Aircraft Configuration

Program Family
[ A320 ]

ATA Chapter
[ 32-42 Braking System ]

Document Family
[x] AMM  [x] IPC  [x] WDM  [x] FIM

Effectivity
[ MSN 001-200 ]
```

## Demo-Facing Architecture

```text
Aircraft configuration + technician question
        |
        v
Technical publication access layer
        |
        v
Semantic retrieval and effectivity filtering
        |
        v
Relevant controlled-publication excerpts
        |
        v
Gemini answer generator
        |
        v
Grounded answer + safety notes + DMC/FIN citations
```

## Run Locally

```bash
cd /Users/sai/Documents/Codex/2026-05-25/files-mentioned-by-the-user-1159667/airbus_brake_streamlit_poc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your_key_here"
streamlit run app.py
```

On first use, click **Initialize knowledge layer** in the app.

## Run In Colab

Open:

`airbus_brake_streamlit_gemini_poc_colab.ipynb`

Then run each cell. The notebook is self-contained: it writes the app and manual extracts into `/content`, installs dependencies, and launches Streamlit through `localtunnel`.
If `localtunnel` asks for a password, use the IP printed by the tunnel cell.

Recommended key setup:

1. In Colab, open the left sidebar.
2. Go to **Secrets**.
3. Add `GEMINI_API_KEY`.
4. Run the notebook.

## Suggested Questions

- What torque is required for the structural retention bolts, and what sequence should I follow?
- CMS-FAULT-32-42-E12 is shown. What is the first isolation step and possible causes?
- For connector CN-LG32 Pin C, give the signal role, wire identifier, and termination.
- Which actuator part number applies to MSN 001-200 and what O-ring kit is required?
- What pressure must be confirmed before loosening any brake line union?

## Production Notes

This is intentionally scoped to one aircraft area and one part family for POC validation, but the UI abstracts implementation details such as source count, local files, and chunking. For production, connect to approved technical-publication storage, version indexes by manual revision, add identity and audit logging, and keep human-in-the-loop review for maintenance decisions.
