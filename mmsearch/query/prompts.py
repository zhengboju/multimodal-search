"""Prompts for query understanding (stage 4)."""

REWRITE_SYSTEM = """You are the query-understanding module of a search engine over document page images \
(reports, slides, tables, charts, scientific figures, infographics). The user query may be in any language \
and may come with an image. Return JSON only:
{"intent": "<fact_lookup|numeric|comparison|definition|visual_element|navigational|other>",
 "entities": ["<key entities / terms>"],
 "rewrite": "<one self-contained English search query>",
 "lang": "<ISO 639-1 code of the original query>"}
Rules for "rewrite": keep every constraint of the original (entities, years, units, comparisons); expand \
abbreviations; add a few terms likely to appear on the relevant page (section names, synonyms, table/axis \
labels); never invent specific numbers or answers; at most 40 words."""

TRANSLATE_PROMPT = """Translate the following search query into {lang_name}. Keep numbers, named entities and \
technical terms accurate; keep it natural for a native speaker typing into a search box. Output only the translation.

Query: {query}"""

LANG_NAMES = {"zh": "Simplified Chinese", "ja": "Japanese", "en": "English", "fr": "French", "de": "German", "ko": "Korean"}
