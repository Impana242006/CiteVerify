
import streamlit as st
import fitz
import re
import json
import csv
import io
import streamlit.components.v1 as components

from pathlib import Path
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CiteVerify",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

REVIEW_FILE = Path(__file__).with_name("reviews.json")


# ==========================================
# 2. PREMIUM NAVY BLUE DASHBOARD CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #07152e,
        #102a50,
        #172554,
        #1e3a5f
    );
    color: #ffffff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.main-title {
    background: linear-gradient(
        110deg,
        #0f172a,
        #1e40af,
        #2563eb
    );
    color: white;
    text-align: center;
    padding: 30px;
    border-radius: 22px;
    font-size: 45px;
    font-weight: 900;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
}

.subtitle {
    text-align: center;
    color: #bfdbfe;
    font-size: 18px;
    font-weight: 600;
    margin: 15px 0 25px 0;
}

h1, h2, h3 {
    color: #bfdbfe !important;
    font-weight: 800 !important;
}

h4 {
    color: #1e3a8a !important;
}

p, label, .stMarkdown {
    color: inherit;
}

div[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #ffffff,
        #dbeafe
    );
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #93c5fd;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.18);
}

div[data-testid="stMetricLabel"] {
    color: #1e40af !important;
    font-weight: bold;
}

div[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 900;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #020617,
        #0f172a,
        #1e3a8a
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.stButton button,
.stDownloadButton button {
    background: linear-gradient(
        90deg,
        #1d4ed8,
        #2563eb,
        #3b82f6
    );
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background: #1e40af;
    transform: translateY(-2px);
}

div[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 2px dashed #60a5fa;
    border-radius: 18px;
    padding: 18px;
    color: #0f172a;
}

div[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #93c5fd;
    border-radius: 14px;
    margin-bottom: 10px;
    color: #0f172a;
}

.stTextArea textarea {
    border-radius: 12px;
    border: 1px solid #60a5fa;
    background: #ffffff;
    color: #0f172a;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

hr {
    border-color: #60a5fa;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. REVIEW HISTORY FUNCTIONS
# ==========================================

def load_reviews():

    try:

        if REVIEW_FILE.exists():

            with open(
                REVIEW_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

                if isinstance(data, list):
                    return data

        return []

    except (json.JSONDecodeError, OSError):

        return []


def save_review(claim, status, notes, filename):

    reviews = load_reviews()

    review = {
        "claim": claim,
        "status": status,
        "notes": notes,
        "filename": filename,
        "date": datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )
    }

    reviews.append(review)

    with open(
        REVIEW_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            reviews,
            file,
            indent=4,
            ensure_ascii=False
        )


# ==========================================
# 4. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🔬 CiteVerify</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Assisted Scientific Claim & Citation Verification System'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Welcome to CiteVerify! Upload a research paper "
    "to explore citation matching, identify claims, "
    "find related evidence and record verification reviews."
)


# ==========================================
# 5. SIDEBAR
# ==========================================

st.sidebar.markdown("# 🔬 CiteVerify")
st.sidebar.markdown("---")

st.sidebar.markdown("### 📌 Dashboard")

st.sidebar.markdown("""
📄 PDF Analysis

🔗 Citation Matching

🧪 Claim Identification

📚 Evidence Finder

📝 Claim Verification

📊 Review Analytics

📥 Download Report
""")

st.sidebar.markdown("---")

st.sidebar.caption("Scientific Research Assistant")
st.sidebar.caption("Version 2.0")


# ==========================================
# 6. PDF UPLOAD
# ==========================================

st.header("📄 Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose your research paper",
    type=["pdf"]
)


if uploaded_file is not None:

    filename = uploaded_file.name

    st.success(
        f"Successfully uploaded: {filename}"
    )

    # ======================================
    # 7. PDF TEXT EXTRACTION
    # ======================================

    st.divider()
    st.header("📄 PDF Text Extraction")

    try:

        with st.spinner("Extracting PDF text..."):

            pdf_document = fitz.open(
                stream=uploaded_file.getvalue(),
                filetype="pdf"
            )

            full_text = ""

            for page in pdf_document:
                full_text += page.get_text() + "\n"

            pdf_document.close()

    except Exception as error:

        st.error(
            f"Unable to read this PDF: {error}"
        )

        full_text = ""

    if not full_text.strip():

        st.warning(
            "No readable text found. "
            "This PDF may contain scanned images."
        )

    else:

        st.success(
            "PDF text extracted successfully!"
        )

        # ======================================
        # 8. VIEW EXTRACTED PDF TEXT
        # ======================================

        st.divider()
        st.header("👁️ View Extracted PDF Text")

        with st.expander(
            "📄 Click here to view extracted PDF text",
            expanded=False
        ):

            st.text_area(
                "Extracted PDF Content",
                value=full_text,
                height=450,
                disabled=True
            )

            st.download_button(
                label="⬇️ Download Extracted Text",
                data=full_text,
                file_name="CiteVerify_Extracted_Text.txt",
                mime="text/plain"
            )

        # ======================================
        # 9. REFERENCE SECTION DETECTION
        # ======================================

        st.divider()
        st.header("📚 Reference Section Detection")

        reference_match = re.search(
            r"\b(references|bibliography|works cited)\b",
            full_text,
            re.IGNORECASE
        )

        if reference_match:

            reference_text = full_text[
                reference_match.start():
            ]

            body_text = full_text[
                :reference_match.start()
            ]

        else:

            reference_text = ""
            body_text = full_text

        # ======================================
        # 10. CITATION MATCHING
        # ======================================

        st.divider()
        st.header("🔗 Citation Matching")

        author_year_citations = re.findall(
            r"\([A-Z][A-Za-z-]+(?:\s+et\s+al\.)?,?\s*(?:19|20)\d{2}[a-z]?\)",
            body_text
        )

        numbered_citations = re.findall(
            r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]",
            body_text
        )

        citation_ids = set(author_year_citations)

        for group in numbered_citations:

            citation_ids.add("[" + group + "]")

        matched = []
        unmatched = []

        for citation in citation_ids:

            if citation.startswith("["):

                number_match = re.search(
                    r"\d+",
                    citation
                )

                number = (
                    number_match.group()
                    if number_match
                    else ""
                )

                if number and re.search(
                    rf"(?m)^\s*\[?{re.escape(number)}\]?",
                    reference_text
                ):

                    matched.append(citation)

                else:

                    unmatched.append(citation)

            else:

                author = re.search(
                    r"[A-Z][A-Za-z-]+",
                    citation
                )

                year = re.search(
                    r"(?:19|20)\d{2}",
                    citation
                )

                if author and year:

                    if (
                        author.group().lower()
                        in reference_text.lower()
                        and year.group()
                        in reference_text
                    ):

                        matched.append(citation)

                    else:

                        unmatched.append(citation)

                else:

                    unmatched.append(citation)

        total_citations = len(citation_ids)
        matched_count = len(matched)
        unmatched_count = len(unmatched)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "📌 Total Citations",
            total_citations
        )

        col2.metric(
            "✅ Matched",
            matched_count
        )

        col3.metric(
            "❌ Unmatched",
            unmatched_count
        )

        if total_citations > 0:

            st.progress(
                matched_count / total_citations
            )

        with st.expander("✅ View Matched Citations"):

            if matched:

                for citation in sorted(matched):
                    st.write("✅", citation)

            else:

                st.write("No matched citations found.")

        with st.expander("❌ View Unmatched Citations"):

            if unmatched:

                for citation in sorted(unmatched):
                    st.write("❌", citation)

            else:

                st.write("No unmatched citations found.")

        # ======================================
        # 11. CLAIM IDENTIFICATION
        # ======================================

        st.divider()
        st.header("🧪 Scientific Claim Identification")

        sentences = re.split(
            r"(?<=[.!?])\s+",
            body_text
        )

        scientific_claims = []

        for sentence in sentences:

            if (
                len(sentence) > 50
                and re.search(
                    r"\([^)]*(?:19|20)\d{2}[^)]*\)|"
                    r"\[\d+(?:\s*[,–-]\s*\d+)*\]",
                    sentence
                )
            ):

                scientific_claims.append(
                    sentence.strip()
                )

        st.metric(
            "Citation-linked Claims Identified",
            len(scientific_claims)
        )

        if scientific_claims:

            for index, claim in enumerate(
                scientific_claims,
                1
            ):

                with st.expander(f"🧪 Claim {index}"):
                    st.write(claim)

        else:

            st.warning(
                "No citation-linked claims were identified."
            )

        # ======================================
        # 12. SCIENTIFIC EVIDENCE FINDER
        # ======================================

        st.divider()
        st.header("📚 Scientific Evidence Finder")

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(
                r"\n+",
                body_text
            )
            if len(paragraph.strip()) > 50
        ]

        if scientific_claims and paragraphs:

            selected_claim = st.selectbox(
                "Select a claim to find related evidence:",
                scientific_claims
            )

            if st.button("🔎 Find Relevant Evidence"):

                try:

                    documents = [
                        selected_claim
                    ] + paragraphs

                    vectorizer = TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2)
                    )

                    tfidf_matrix = vectorizer.fit_transform(
                        documents
                    )

                    similarities = cosine_similarity(
                        tfidf_matrix[0:1],
                        tfidf_matrix[1:]
                    ).flatten()

                    top_indices = similarities.argsort()[
                        -3:
                    ][::-1]

                    st.subheader(
                        "Relevant Evidence Passages and Scores"
                    )

                    for index in top_indices:

                        st.write(
                            f"Similarity Score: "
                            f"{similarities[index]:.2f}"
                        )

                        st.info(paragraphs[index])

                    st.caption(
                        "Similarity scores indicate text overlap. "
                        "They do not prove scientific support "
                        "or contradiction."
                    )

                except ValueError:

                    st.warning(
                        "Not enough usable text to compare."
                    )

        else:

            st.info(
                "Citation-linked claims or evidence "
                "passages are not available."
            )

        # ======================================
        # 13. CLAIM VERIFICATION
        # ======================================

        st.divider()
        st.header("📝 Claim Verification Status")

        if scientific_claims:

            claim_to_review = st.selectbox(
                "Choose a claim to review:",
                scientific_claims,
                key="review_claim"
            )

            with st.form("claim_review_form"):

                st.write("### Your Verification Decision")

                review_status = st.radio(
                    "Select verification status:",
                    [
                        "Needs Review",
                        "Supported",
                        "Contradicted"
                    ]
                )

                review_notes = st.text_area(
                    "Reason for your decision:",
                    placeholder=(
                        "Explain why you selected this status..."
                    )
                )

                submit_review = st.form_submit_button(
                    "💾 Submit Review"
                )

                if submit_review:

                    if not review_notes.strip():

                        st.warning(
                            "Please enter a reason."
                        )

                    else:

                        save_review(
                            claim_to_review,
                            review_status,
                            review_notes,
                            filename
                        )

                        st.success(
                            "Your review has been saved successfully!"
                        )

        else:

            st.info(
                "No citation-linked claims are available for review."
            )


# ==========================================
# 14. REVIEW ANALYTICS DASHBOARD
# ==========================================

st.divider()

st.header("📊 Review Analytics Dashboard")

reviews = load_reviews()

total_reviews = len(reviews)

supported_count = sum(
    1 for review in reviews
    if review.get("status") == "Supported"
)

contradicted_count = sum(
    1 for review in reviews
    if review.get("status") == "Contradicted"
)

needs_review_count = sum(
    1 for review in reviews
    if review.get("status") == "Needs Review"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("📋 Total Reviews", total_reviews)
col2.metric("🟢 Supported", supported_count)
col3.metric("🔴 Contradicted", contradicted_count)
col4.metric("🟡 Needs Review", needs_review_count)


# ==========================================
# 15. CUSTOM ANALYTICS - NO PANDAS
# ==========================================

if total_reviews > 0:

    st.subheader("📈 Verification Overview")

    supported_percentage = (
        supported_count / total_reviews
    ) * 100

    contradicted_percentage = (
        contradicted_count / total_reviews
    ) * 100

    needs_review_percentage = (
        needs_review_count / total_reviews
    ) * 100

    st.markdown(f"""
    <div style="
        background:white;
        padding:22px;
        border-radius:18px;
        border:1px solid #ddd6fe;
        margin-bottom:15px;
    ">
        <h4 style="color:#312e81;">🟢 Supported Reviews</h4>
        <h2 style="color:#16a34a;">{supported_percentage:.1f}%</h2>
        <div style="background:#e5e7eb;border-radius:20px;height:16px;">
            <div style="
                background:#16a34a;
                width:{supported_percentage}%;
                height:16px;
                border-radius:20px;
            "></div>
        </div>
        <p>{supported_count} reviews</p>
    </div>

    <div style="
        background:white;
        padding:22px;
        border-radius:18px;
        border:1px solid #ddd6fe;
        margin-bottom:15px;
    ">
        <h4 style="color:#312e81;">🔴 Contradicted Reviews</h4>
        <h2 style="color:#dc2626;">{contradicted_percentage:.1f}%</h2>
        <div style="background:#e5e7eb;border-radius:20px;height:16px;">
            <div style="
                background:#dc2626;
                width:{contradicted_percentage}%;
                height:16px;
                border-radius:20px;
            "></div>
        </div>
        <p>{contradicted_count} reviews</p>
    </div>

    <div style="
        background:white;
        padding:22px;
        border-radius:18px;
        border:1px solid #ddd6fe;
        margin-bottom:15px;
    ">
        <h4 style="color:#312e81;">🟡 Needs Review</h4>
        <h2 style="color:#ca8a04;">{needs_review_percentage:.1f}%</h2>
        <div style="background:#e5e7eb;border-radius:20px;height:16px;">
            <div style="
                background:#eab308;
                width:{needs_review_percentage}%;
                height:16px;
                border-radius:20px;
            "></div>
        </div>
        <p>{needs_review_count} reviews</p>
    </div>
    """, unsafe_allow_html=True)

    # ======================================
    # PIE CHART - REVIEW STATUS DISTRIBUTION
    # ======================================

    st.divider()
    st.subheader("🥧 Review Status Distribution")

    pie_html = f"""
    <html>
    <body style="
        margin:0;
        background:transparent;
        font-family:Arial,sans-serif;
    ">
        <div style="
            display:flex;
            justify-content:center;
            align-items:center;
            gap:35px;
            flex-wrap:wrap;
            padding:20px;
        ">

            <div style="
                width:230px;
                height:230px;
                border-radius:50%;
                background:conic-gradient(
                    #16a34a 0% {supported_percentage}%,
                    #dc2626 {supported_percentage}% {supported_percentage + contradicted_percentage}%,
                    #eab308 {supported_percentage + contradicted_percentage}% 100%
                );
                box-shadow:0 8px 25px rgba(0,0,0,0.15);
            "></div>

            <div style="
                font-size:17px;
                line-height:2.5;
                color:#ffffff;
                font-weight:bold;
            ">
                <div>
                    <span style="color:#22c55e;">●</span>
                    Supported: {supported_count}
                </div>

                <div>
                    <span style="color:#f87171;">●</span>
                    Contradicted: {contradicted_count}
                </div>

                <div>
                    <span style="color:#facc15;">●</span>
                    Needs Review: {needs_review_count}
                </div>
            </div>

        </div>
    </body>
    </html>
    """

    components.html(
        pie_html,
        height=290,
        scrolling=False
    )

    st.subheader("🎯 Supported Review Percentage")

    st.progress(
        supported_count / total_reviews
    )

    st.write(
        f"Supported reviews: {supported_percentage:.1f}%"
    )

else:

    st.info(
        "Submit your first claim review to see analytics."
    )


# ==========================================
# 16. REVIEW HISTORY
# ==========================================

st.divider()

st.header("📋 Review History")

if reviews:

    st.write(
        f"Total saved reviews: {total_reviews}"
    )

    for index, review in enumerate(
        reversed(reviews),
        1
    ):

        with st.expander(
            f"Review {index} - "
            f"{review.get('status', 'Unknown')}"
        ):

            st.write("**Scientific Claim:**")
            st.write(review.get("claim", ""))

            st.write("**Verification Status:**")
            st.write(review.get("status", ""))

            st.write("**Reason:**")
            st.write(review.get("notes", ""))

            st.write("**PDF Filename:**")
            st.write(review.get("filename", ""))

            st.write("**Review Date:**")
            st.write(review.get("date", ""))

    # ======================================
    # 17. DOWNLOAD CSV REPORT
    # ======================================

    st.divider()

    st.header("📥 Download Review Report")

    csv_buffer = io.StringIO()

    fieldnames = [
        "claim",
        "status",
        "notes",
        "filename",
        "date"
    ]

    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=fieldnames,
        extrasaction="ignore"
    )

    writer.writeheader()

    writer.writerows(reviews)

    csv_data = csv_buffer.getvalue().encode(
        "utf-8-sig"
    )

    st.download_button(
        label="⬇️ Download CiteVerify Report",
        data=csv_data,
        file_name="CiteVerify_Review_Report.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No reviews available yet. "
        "Submit a claim review to see it here."
    )


# ==========================================
# 18. FOOTER
# ==========================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    padding:25px;
    background:linear-gradient(
        90deg,
        #0f172a,
        #1e40af,
        #2563eb
    );
    border-radius:18px;
    color:white;
">
    <h2 style="color:white !important;">
        🔬 CiteVerify
    </h2>
    Scientific Claim & Citation Verification System
    <br>
    <small>Research • Evidence • Verification</small>
</div>
""", unsafe_allow_html=True)

st.caption(
    "CiteVerify provides preliminary citation matching "
    "and text-similarity analysis. It does not establish "
    "scientific truth. Human verification is required."
)
