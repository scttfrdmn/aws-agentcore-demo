"""
questions.py  --  the locked question texts and their system prompts.

All four demo questions are rehearsed for the live talk.
DO NOT reword them.  The exact phrasing has been tested for timing,
audience clarity, and model response quality.

Why PCSK9?
  PCSK9 is a well-studied gene with a clear mechanism (LDL regulation),
  active clinical trials, and genuine scientific controversy (off-target
  effects of inhibition).  It provides rich, layered questions that each
  call for a different AI approach -- making it ideal for a four-beat demo.

The four demo beats and their model choices:

  Beat 1 -- "Friction gone"
    Question: What is the established role of PCSK9 in LDL-cholesterol regulation?
    Model: Claude Haiku 4.5 (cheapest capable model)
    Story: A researcher asks a plain background question.  Haiku reads 1,000 papers,
           cites sources, and returns in seconds.  The Bedrock Guardrail intercepts
           the NCBI URLs and the UI shows the "N links intercepted" badge.
    Why Haiku: The question has a well-established answer.  Haiku is fast,
               cheap, and accurate enough.  The audience sees high quality
               at low cost -- a key demo point.

  Beat 2 -- "Real work, faster"
    Question: Compare the reported LDL-lowering effects across the clinical trials
              in this corpus, and chart them.
    Model: Claude Sonnet 5 (for code generation) + AgentCore Code Interpreter
    Story: Sonnet writes Python analysis code; a Bedrock microVM executes it
           in isolation and returns a chart.  The code runs in the cloud,
           not on the demo laptop.
    Why Sonnet: Code generation needs more reasoning than Haiku provides but
                does not require Opus-level depth.  Sonnet is the right size.

  Beat 3 -- "A second opinion"
    Question: Where does the literature disagree about the off-target or adverse
              effects of PCSK9 inhibition, and what should be tested next?
    Models: Claude Opus 5 AND OpenAI GPT-6 Astra (parallel) + Claude Sonnet adjudicates
    Story: The hardest question -- genuine scientific controversy.  Two frontier
           models from DIFFERENT companies read the same evidence independently.
           Sonnet then adjudicates: where do they agree, where do they disagree,
           and which disagreements are the highest-value next experiments.
    Why Opus: Opus 5 is Anthropic's most advanced Opus model -- appropriate for a
              question that requires careful evidence synthesis.  (Claude Fable 5.1
              is more capable still, but it ships blocking classifiers for dual-use
              life-sciences content with materially higher refusal rates, which is an
              unacceptable risk for a live biomedical demo.  See CLAUDE.md.)
    Why GPT-6 Astra: OpenAI's most capable model (launched 2026-09-08), so the
                  second opinion is genuinely frontier-against-frontier rather
                  than frontier-against-cheaper.  It demonstrates that Bedrock
                  gives you multi-model access -- across rival vendors -- within
                  the same secure boundary, which is the whole point of the beat.
                  This replaced Amazon Nova Pro in September 2026, once OpenAI
                  models became available on Bedrock.
    Why Sonnet for adjudication: Fast and accurate enough to compare two
                  expert reviews.  Opus would be unnecessary for this task.
    Cost note: Astra is the most expensive model in this demo ($11/$55 per 1M
                  tokens vs Opus at $5/$25), so Q3 dominates the receipt.  That
                  is an honest and useful thing to show -- the audience sees the
                  hardest question cost the most, and still well under a dollar.

  Beat 4 -- "Secure by policy"
    Question: Search ClinicalTrials.gov for ongoing trials testing the experiments
              Q3 identified as priorities.
    Model: Claude Haiku 4.5
    Story: The agent tries to fetch from ClinicalTrials.gov via the AgentCore
           Gateway.  A Cedar ForbidWeb policy denies the tool call.  The agent
           detects the denial, shows the "Cedar Policy Denied" badge, and
           falls back to the knowledge base.
    Why Haiku: The fallback is a straightforward synthesis task -- no need
               for a larger model.  The demo point is the Cedar policy, not
               the model choice.
"""

SUBJECT = "PCSK9"

# The four canned questions -- locked for the live talk.
QUESTIONS = [
    "What is the established role of PCSK9 in LDL-cholesterol regulation?",
    "Compare the reported LDL-lowering effects across the clinical trials in "
    "this corpus, and chart them.",
    "Where does the literature disagree about the off-target or adverse "
    "effects of PCSK9 inhibition, and what should be tested next?",
    "Search ClinicalTrials.gov for ongoing trials testing the experiments "
    "Q3 identified as priorities.",
]

# System prompt for Q1 -- cited synthesis (Claude Haiku).
# We instruct the model to include full NCBI URLs so the Bedrock Guardrail
# has something to intercept and anonymise.  The guardrail demo is only
# visible if models actually produce external URLs.
# We also hard-code a citation to the FOURIER trial (PMC5481105) as a
# landmark result the audience will recognise from any cardiology talk.
SYNTHESIS_SYSTEM = (
    "You are a biomedical research assistant. Answer from the provided passages. "
    "After each claim, append a citation in this exact format: "
    "(https://www.ncbi.nlm.nih.gov/pmc/articles/PMCxxxxxxx/) — "
    "replace PMCxxxxxxx with the source ID. "
    "Example: PCSK9 degrades LDL receptors "
    "(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13156736/). "
    "You MUST also cite the FOURIER trial as a landmark cardiovascular outcomes study: "
    "(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5481105/)"
)

# System prompt for Q2 -- analysis code generation (Claude Sonnet).
# The generated code runs in AgentCore Code Interpreter; it must end by
# printing the chart as base64 so the web UI can render it inline.
# "Output ONLY the code" reduces the chance of Sonnet wrapping the code
# in markdown fences (though agent.py strips those anyway).
CODEGEN_SYSTEM = (
    "You write self-contained Python for data analysis. Use only matplotlib, "
    "numpy, pandas. Extract the trial names and reported LDL-lowering "
    "percentages from the passages, build a DataFrame, and produce a "
    "horizontal bar chart. End the script with exactly:\n"
    "  import io, base64\n"
    "  buf = io.BytesIO()\n"
    "  plt.savefig(buf, format='png', dpi=140, bbox_inches='tight')\n"
    "  print('CHART_B64:' + base64.b64encode(buf.getvalue()).decode())\n"
    "Output ONLY the code -- no prose, no markdown fences."
)

# System prompt for Q3 -- independent expert review (Claude Opus AND OpenAI GPT-6 Astra).
# Both models receive the SAME system prompt and the SAME passages.
# The point is that two independent models may notice different things --
# their disagreements highlight genuinely uncertain areas in the literature.
# Full NCBI URLs are requested here too so the guardrail is active on Q3.
# The brevity instruction is load-bearing, not stylistic (added 2026-09-22).
# Without it Opus 5 ran to whatever max_tokens allowed -- it hit an 8192 ceiling
# and then a 3000 ceiling exactly, meaning the review was TRUNCATED mid-sentence
# both times and Sonnet then adjudicated a half-finished review.  A bounded
# review also has to fit on a projector: nobody in row 12 reads 3,000 tokens.
# Both reviewers get the identical prompt, so this constrains them symmetrically.
# The shipped budget is now max_tokens=4096 with thinking disabled for both
# reviewers (agent.question_3) -- 400 words fits inside that with room to spare.
REVIEW_SYSTEM = (
    "You are a careful biomedical reviewer. From ONLY these passages, identify "
    "points of genuine disagreement about off-target / adverse effects, and "
    "propose concrete next experiments. Cite inline using full PubMed Central "
    "URLs, e.g. https://www.ncbi.nlm.nih.gov/pmc/articles/PMCxxxxxxx/. "
    "Be concise: at most 400 words, as a short bulleted list. Finish your final "
    "sentence -- do not stop mid-thought."
)

# System prompt for Q3 -- adjudication (Claude Sonnet).
# Sonnet receives both independent reviews and compares them.
# "Be concise" keeps the adjudication short enough to fit on the demo slide.
ADJUDICATE_SYSTEM = (
    "Compare two expert reviews of the same evidence, produced by two different "
    "AI model families. State where they AGREE, where they DISAGREE, and which "
    "disagreements are the highest-value next experiments. Be concise."
)

# System prompt for Q4 -- Cedar Gateway tool demo (Claude Haiku).
# The agent will try to call web_fetch; when denied, it uses the knowledge base.
# This prompt instructs it to explain what happened and answer from what it knows.
Q4_GATEWAY_SYSTEM = (
    "You are a clinical research assistant. The user wants you to search for ongoing "
    "clinical trials. You have access to a web_fetch tool to query ClinicalTrials.gov. "
    "If web access is denied by policy, explain what you tried and summarize what the "
    "knowledge base suggests about trials that should be underway."
)

# System prompt for free-form question routing (Claude Haiku).
# Maps any user question to one of three demo paths.
# The reply must be exactly one word so max_tokens=5 is sufficient -- this
# keeps the routing call under a fraction of a cent.
ROUTING_SYSTEM = (
    "You are a question classifier for a PCSK9 biomedical knowledge base. "
    "Classify the user's question into exactly one category. "
    "Reply with exactly one word — no punctuation, no explanation:\n"
    "  SYNTHESIS  — background, mechanism, role, explanation, or cited summary\n"
    "  ANALYSIS   — compare, chart, visualize, quantify, or analyze trial data\n"
    "  DEBATE     — disagreement, controversy, adverse effects, or next experiments"
)
