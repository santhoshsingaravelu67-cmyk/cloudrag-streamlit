"""Verify copied evidence and attach citations from the prepared source excerpts.

This checks textual provenance, not relevance, completeness, or whether a source
is true. It does not validate generated paraphrases. Whitespace may differ, but
words, capitalization, internal punctuation, negation and numbers must match.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


MAX_SELECTIONS = 4
MAX_ANSWER_CHARS = 20_000
MAX_SOURCE_CHARS = 100_000
MAX_EXCERPTS = 20
_CITATION = re.compile(r"\[S[0-9]+\]")
_BULLET = re.compile(r"(?:[-*•]\s+|[0-9]{1,2}[.)]\s+)")
_ENDING = re.compile(r"[.!?](?:[\"”’')\]]*)(?=\s|$)")
_ABBREVIATIONS = re.compile(r"(?:\b(?:dr|mr|mrs|ms|prof|sr|jr|st|vs|etc)|\be\.g|\bi\.e|\bu\.s|\bu\.k)\.$", re.I)
_INVALID_FORMAT = "The AI selection format was invalid and was withheld."
_UNMATCHED = "The AI selection could not be verified as copied source text and was withheld."


@dataclass(frozen=True)
class _Sentence:
    start: int
    end: int
    complete: bool


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _start_of_text(text: str, start: int, end: int) -> int:
    while start < end and text[start].isspace():
        start += 1
    marker = _BULLET.match(text, start)
    return marker.end() if marker and marker.end() < end else start


def _sentence_ranges(text: str) -> list[_Sentence]:
    """Find practical English sentence boundaries without splitting decimals.

    A final unfinished excerpt remains incomplete and cannot supply a quote.
    Common title abbreviations and initials do not terminate a sentence.
    """
    sentences = []
    start = _start_of_text(text, 0, len(text))
    for ending in _ENDING.finditer(text):
        if ending.start() < start:
            continue
        prefix = text[max(start, ending.start() - 12):ending.start() + 1]
        if text[ending.start()] == "." and (
            _ABBREVIATIONS.search(prefix) or re.search(r"(?:^|\s)[A-Z]\.$", prefix)
        ):
            continue
        begin = _start_of_text(text, start, ending.end())
        if begin < ending.end():
            sentences.append(_Sentence(begin, ending.end(), True))
        start = _start_of_text(text, ending.end(), len(text))
    begin = _start_of_text(text, start, len(text))
    if begin < len(text):
        sentences.append(_Sentence(begin, len(text), False))
    return sentences


def _source_sentences(original: str) -> tuple[str, list[_Sentence]]:
    text = _normalise(original)
    sentences = []
    cursor = 0
    # Paragraphs prevent a heading or an unfinished chunk prefix from being
    # prepended to the first complete sentence in the following paragraph.
    for paragraph in re.split(r"\n\s*\n", original):
        normalised = _normalise(paragraph)
        if not normalised:
            continue
        position = text.find(normalised, cursor)
        if position < 0:
            raise RuntimeError("The prepared source excerpts are invalid.")
        cursor = position + len(normalised)
        lines = paragraph.splitlines()
        while lines and (not lines[0].strip() or re.match(r"^\s*#{1,6}\s", lines[0])):
            lines.pop(0)
        body = _normalise("\n".join(lines))
        if not body:
            continue
        body_offset = position + len(normalised) - len(body)
        sentences.extend(_Sentence(body_offset + sentence.start, body_offset + sentence.end,
                                   sentence.complete)
                         for sentence in _sentence_ranges(body))
    return text, sentences


def _quoted_selections(answer: str) -> list[str]:
    selections = []
    position = 0
    while position < len(answer):
        while position < len(answer) and answer[position].isspace():
            position += 1
        if position == len(answer):
            break
        marker = _BULLET.match(answer, position)
        if marker:
            position = marker.end()
        if position >= len(answer) or answer[position] not in {'"', '“'}:
            raise RuntimeError(_INVALID_FORMAT)
        closer = '"' if answer[position] == '"' else '”'
        end = answer.find(closer, position + 1)
        if end < 0:
            raise RuntimeError(_INVALID_FORMAT)
        selection = answer[position + 1:end]
        if not selection.strip():
            raise RuntimeError(_INVALID_FORMAT)
        selections.append(selection)
        position = end + 1
        # Accept punctuation outside the wrapper, as well as inside the quote.
        if position < len(answer) and answer[position] in ".!?":
            position += 1
        whitespace_start = position
        while position < len(answer) and answer[position].isspace():
            position += 1
        citation = _CITATION.match(answer, position)
        if citation:
            position = citation.end()
            if position < len(answer) and answer[position] in ".!?":
                position += 1
            if position < len(answer) and not answer[position].isspace():
                raise RuntimeError(_INVALID_FORMAT)
        elif position < len(answer) and position == whitespace_start:
            raise RuntimeError(_INVALID_FORMAT)
        if len(selections) > MAX_SELECTIONS:
            raise RuntimeError(_INVALID_FORMAT)
    return selections


def _plain_selections(answer: str) -> list[str]:
    selections = []
    cursor = 0
    pieces = []
    for citation in _CITATION.finditer(answer):
        piece = answer[cursor:citation.start()].strip()
        if not piece:
            raise RuntimeError(_INVALID_FORMAT)
        pieces.append(piece)
        cursor = citation.end()
        if cursor < len(answer) and answer[cursor] in ".!?":
            cursor += 1
    if answer[cursor:].strip():
        pieces.append(answer[cursor:].strip())
    for piece in pieces:
        if re.search(r"\[(?:S|Source)", piece) or piece.startswith(('"', '“', '”')):
            raise RuntimeError(_INVALID_FORMAT)
        text = _normalise(piece)
        for sentence in _sentence_ranges(text):
            selections.append(text[sentence.start:sentence.end])
            if len(selections) > MAX_SELECTIONS:
                raise RuntimeError(_INVALID_FORMAT)
    return selections


def _selections(answer: str) -> list[str]:
    if (not isinstance(answer, str) or not answer.strip() or len(answer) > MAX_ANSWER_CHARS
            or any(ord(char) < 32 and char not in "\n\r\t" for char in answer)
            or any(delimiter in answer for delimiter in ("<|", "|>", "```", "~~~"))):
        raise RuntimeError(_INVALID_FORMAT)
    beginning = answer.lstrip()
    marker = _BULLET.match(beginning)
    if marker:
        beginning = beginning[marker.end():]
    selections = (_quoted_selections(answer) if beginning.startswith(('"', '“', '”'))
                  else _plain_selections(answer))
    if not selections or len(selections) > MAX_SELECTIONS:
        raise RuntimeError(_INVALID_FORMAT)
    return selections


def _match_text(selection: str) -> str:
    text = _normalise(selection)
    marker = _BULLET.match(text)
    if marker:
        text = text[marker.end():]
    # Only terminal punctuation is optional; internal punctuation remains exact.
    return text.rstrip(".!?;:,").rstrip()


def _enclosing_quote(selection: str, text: str, sentences: list[_Sentence]) -> str | None:
    match_text = _match_text(selection)
    if not match_text:
        return None
    pattern = re.escape(match_text)
    if re.match(r"\w", match_text[0]):
        pattern = r"(?<!\w)" + pattern
    if re.match(r"\w", match_text[-1]):
        pattern += r"(?!\w)"
    word_count = len(re.findall(r"\b\w+(?:[-’']\w+)*\b", match_text))
    candidates = []
    for match in re.finditer(pattern, text):
        covering = [sentence for sentence in sentences
                    if sentence.start < match.end() and sentence.end > match.start()]
        if (not covering or any(not sentence.complete for sentence in covering)
                or covering[0].start > match.start() or covering[-1].end < match.end()):
            continue
        if any(text[left.end:right.start].strip() for left, right in zip(covering, covering[1:])):
            continue
        start, end = covering[0].start, covering[-1].end
        quote = text[start:end]
        if any(delimiter in quote for delimiter in ("<|", "|>", "```", "~~~")):
            continue
        if word_count < 4 and match_text != _match_text(quote):
            continue
        candidates.append((end - start, start, quote))
    return min(candidates)[2] if candidates else None


def ground_quotes(answer: str, excerpts: list[dict]) -> str:
    """Return copied, sentence-expanded quotes with server-assigned citations.

    Every selection must match contiguous text in the provided prepared excerpts.
    Model-written citation IDs are ignored. A partial selection needs at least
    four words and expands to its enclosing complete source sentence(s). The
    first matching source wins; within it, the smallest complete span wins.
    Invalid or unsupported material withholds the whole answer. This guarantees
    copied provenance only, never relevance, completeness or factual truth.
    """
    selections = _selections(answer)
    if (not isinstance(excerpts, list) or not excerpts or len(excerpts) > MAX_EXCERPTS
            or any(not isinstance(entry, dict) for entry in excerpts)):
        raise RuntimeError("The prepared source excerpts are invalid.")
    prepared = []
    seen_ids = set()
    for entry in excerpts:
        citation, original = entry.get("id"), entry.get("text")
        if (not isinstance(citation, str) or not re.fullmatch(r"S[1-9][0-9]*", citation)
                or citation in seen_ids or not isinstance(original, str) or not original.strip()
                or len(original) > MAX_SOURCE_CHARS):
            raise RuntimeError("The prepared source excerpts are invalid.")
        seen_ids.add(citation)
        text, sentences = _source_sentences(original)
        prepared.append((citation, text, sentences))
    output = []
    seen_quotes = set()
    for selection in selections:
        for citation, text, sentences in prepared:
            quote = _enclosing_quote(selection, text, sentences)
            if quote is not None:
                if quote not in seen_quotes:
                    output.append(f'"{quote}" [{citation}]')
                    seen_quotes.add(quote)
                break
        else:
            raise RuntimeError(_UNMATCHED)
    return "\n".join(output)
