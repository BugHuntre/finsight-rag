from finsight.prompts import NOT_FOUND_MESSAGE, build_messages, format_context, page_label
from finsight.retriever import RetrievedChunk


def _chunk(
    company="TCS",
    document="Annual Report FY2026",
    page=10,
    printed_page=0,
    text="Revenue grew.",
):
    return RetrievedChunk(
        text=text,
        metadata={
            "company": company,
            "document": document,
            "page": page,
            "printed_page": printed_page,
        },
        distance=0.2,
    )


def test_page_label_prefers_printed_page():
    assert page_label({"page": 45, "printed_page": 75}) == 75


def test_page_label_falls_back_to_pdf_page_when_unknown():
    assert page_label({"page": 45, "printed_page": 0}) == 45


def test_format_context_numbers_blocks_from_one():
    chunks = [_chunk(company="TCS", text="First."), _chunk(company="Infosys", text="Second.")]
    context = format_context(chunks)
    assert context.startswith("[1] TCS")
    assert "[2] Infosys" in context
    assert "First." in context and "Second." in context


def test_format_context_uses_printed_page_in_label():
    context = format_context([_chunk(page=45, printed_page=75)])
    assert "(page 75)" in context


def test_build_messages_includes_system_prompt_rules():
    messages = build_messages("What was revenue?", [_chunk()])
    assert messages[0]["role"] == "system"
    assert NOT_FOUND_MESSAGE in messages[0]["content"]
    assert "Cite every factual claim" in messages[0]["content"]


def test_build_messages_embeds_context_and_question():
    messages = build_messages("What was revenue?", [_chunk(text="Revenue was Rs 100 crore.")])
    user = messages[1]["content"]
    assert "Revenue was Rs 100 crore." in user
    assert "Question: What was revenue?" in user


def test_build_messages_with_no_chunks_says_so_without_fabricating_context():
    messages = build_messages("What was revenue?", [])
    user = messages[1]["content"]
    assert "No relevant context was found" in user
    assert "[1]" not in user
