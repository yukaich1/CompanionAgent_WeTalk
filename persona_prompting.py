from textwrap import dedent


def build_persona_summary_prompt(persona_name, source_label, source_text, reference_text):
    return dedent(
        f"""
        You are extracting a reusable roleplay/persona canon from source material.

        Focus on:
        1. speech style
        2. catchphrases and recurring signature expressions
        3. addressing habits and self-reference habits
        4. sentence endings or recurring turn-of-phrase details
        5. personality
        6. values
        7. worldview
        8. likes/dislikes
        9. appearance
        10. core setup
        11. life experiences only as natural conversational hooks

        Important:
        - Do not assume catchphrases are automatically covered by speech style. Extract them explicitly when present.
        - If the material shows repeated nicknames, self-address, habitual ways of calling others, or signature sentence endings, capture them explicitly.
        - When extracting display_keywords, prioritize the character's most central persona tags: core temperament, value orientation, archetype, role identity, and memorable defining traits.
        - Good display_keywords look like concise persona labels such as “腹黑”, “毒舌”, “旅人”, “魔女”, “理想主义”, “冷静”, “守序”.
        - Avoid display_keywords that are scene-specific phrases, plot fragments, long clauses, audience commentary, or generic descriptive sentences.
        - Prefer 4 to 6 highly representative persona tags over many weak ones.
        - display_keywords should emphasize personality, speech feel, values, role identity, and memorable defining traits first.
        - catchphrases, addressing_habits, sentence_endings, and other learned details are still important, but they belong in their own fields instead of being forced into display_keywords.
        - display_keywords are only high-level UI tags and retrieval hints. They must not replace the richer persona details found elsewhere in the summary.
        - Reject fandom commentary, audience reactions, popularity notes, production notes, and author-side notes.
        - If the text describes what audiences, readers, fans, viewers, outsiders, or commentators like about the character, that is NOT the character's own preference.
        - Never turn external commentary into persona canon.
        - Capture stable likes, dislikes, aversions, preferences, and important lived experiences from the material when they are actually supported.
        - Do not invent story events.
        - For story, background, or life experience details, trust the provided source material first and use public reference snippets only as support.
        - Public reference snippets are supplements only. They must not overwrite, delete, or weaken supported facts from the user-provided material.
        - If the source material and public references do not clearly support an event, relationship, timeline point, or background detail, leave it out.
        - Return short concrete Chinese phrases.
        - display_keywords must be short UI-friendly persona tags, not sentences.
        - catchphrases should prefer short original recurring expressions from the character.
        - addressing_habits should capture how the character refers to self, user, or others.
        - sentence_endings should capture signature particles, endings, or habitual closing style if clearly present.

        Character name: {persona_name}
        Source label: {source_label}

        Source material:
        ```text
        {source_text}
        ```

        Optional public reference snippets:
        ```text
        {reference_text}
        ```
        """
    ).strip()
