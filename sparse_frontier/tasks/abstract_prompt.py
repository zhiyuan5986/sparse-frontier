# Base template with shared components
_BASE_TEMPLATE = """{task_intro}

{question_intro}

{question_section}

<context>
{context}
</context>

{question_repeat_section}

Instructions:
1. First, provide a brief explanation of your reasoning process. Explain how you identified 
   the relevant information from the context and how you determined your answer.
2. Then, provide your final answer following this exact format:
<answer>
{answer_format}
</answer>

Your response must follow this structure exactly:
<explanation>
Your explanation here...
</explanation>
<answer>
Your answer here...
</answer>

Important:
{extra_instructions}
- Keep your explanations clear, coherent, concise, and to the point.
- Do not include any additional text, explanations, or reasoning in the answer section. Follow the answer format exactly.
"""

# Template for single question tasks
SINGLEQ_PROMPT_TEMPLATE = _BASE_TEMPLATE.format(
    task_intro="{task_intro}",
    context="{context}",
    question_intro="Below is your question. I will state it both before and after the context.",
    question_section="<question>\n{question}\n</question>",
    question_repeat_section="<question_repeated>\n{question}\n</question_repeated>",
    answer_format="{answer_format}",
    extra_instructions="{extra_instructions}"
)

# Template for multiple question tasks
MULTIPLEQ_PROMPT_TEMPLATE = _BASE_TEMPLATE.format(
    task_intro="{task_intro}",
    context="{context}",
    question_intro="Below are your questions. I will state them both before and after the context.",
    question_section="<questions>\n{question}\n</questions>",
    question_repeat_section="<questions_repeated>\n{question}\n</questions_repeated>",
    answer_format="{answer_format}",
    extra_instructions="{extra_instructions}"
)
