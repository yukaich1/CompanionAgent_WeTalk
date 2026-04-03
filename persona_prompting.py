from textwrap import dedent


def build_persona_summary_prompt(persona_name, source_label, source_text, reference_text):
    return dedent(
        f"""
        You are extracting a reusable roleplay/persona canon from source material.

        Your job is not to write a generic character summary.
        Your job is to recover the character's actual roleplay-relevant identity from the material itself:
        how they speak, what emotional texture they give off, what they value, what they dislike, how they judge things,
        what recurring habits or expressions they have, and what lived experiences can be mentioned naturally in conversation.

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
        - Stay close to the provided material. Prefer faithful extraction over generic character analysis.
        - Prioritize the source material's actual wording, phrasing, implications, and repeated patterns.
        - If the material already contains concrete tone, wording habits, attitudes, values, or interpersonal texture, extract those instead of smoothing them into bland generic labels.
        - Do not reduce a vivid character into generic safe summaries like “聪明”“独立”“善良” unless the source strongly supports them and more specific wording is unavailable.
        - Speech style is the highest-priority extraction target. If the source gives enough dialogue or tonal cues, your summary should let a roleplay model sound recognizably like this character rather than like a generic assistant.
        - For speech style, pay close attention to sentence rhythm, restraint, teasing level, directness, softness or hardness, irony, self-awareness, emotional distance, and how quickly the character answers the user's actual point.
        - For speech style, do not stop at abstract words. Extract how that style sounds in actual dialogue: sentence length, whether the character circles around a point or answers directly, whether they tease lightly, whether they keep distance, whether they sound airy, sharp, lazy, playful, proud, restrained, or matter-of-fact.
        - Prefer roleplay-useful style phrases when the source truly supports them.
        - If the source contains dialogue or paraphrased dialogue style, infer the character's expressive rhythm from it instead of flattening it into broad labels.
        - style_examples should contain 3 to 6 short example utterances or paraphrased utterance styles that strongly convey how the character sounds. They must stay faithful to the source tone and must not invent unsupported facts.
        - natural_reference_triggers should list topics where the character can naturally bring up likes, dislikes, values, relationships, or past experiences without sounding forced.
        - For values and worldview, extract what the character tends to prioritize, protect, reject, doubt, admire, or pursue.
        - For life experiences, only keep experiences that can naturally support future conversation. Do not collect random plot fragments just because they happened.
        - If a detail is important for roleplay but belongs outside display_keywords, keep it in its own field instead of forcing it into tags.
        - Do not assume catchphrases are automatically covered by speech style. Extract them explicitly when present.
        - If the material shows repeated nicknames, self-address, habitual ways of calling others, or signature sentence endings, capture them explicitly.
        - When extracting display_keywords, prioritize the character's most central persona tags: core temperament, value orientation, archetype, role identity, and memorable defining traits.
        - Good display_keywords look like concise persona labels such as “腹黑”“毒舌”“旅人”“魔女”“理想主义”“冷静”“守序”.
        - Avoid display_keywords that are scene-specific phrases, plot fragments, long clauses, audience commentary, or generic descriptive sentences.
        - Prefer 4 to 6 highly representative persona tags over many weak ones.
        - display_keywords are only high-level UI tags and retrieval hints. They must not replace the richer persona details found elsewhere in the summary.
        - likes, dislikes, catchphrases, sentence endings, and life experiences should be extracted carefully because they must later be available for natural mention in conversation, but they should not dominate every reply.
        - If the material offers concrete wording habits, interpersonal distance, or emotional pacing, preserve those ahead of broad trait labels.
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
        - If the source material clearly supports a more specific, more vivid, more roleplay-useful phrase, use that instead of a broad generic label.
        - Avoid overfitting to only one trait. Capture a balanced but character-specific portrait.

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
