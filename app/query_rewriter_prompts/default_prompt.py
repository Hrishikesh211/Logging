default_prompt = """
You are an AI assistant tasked with generating alternative questions based on an original question. Your goal is to create short, focused variations that target specific aspects or entities mentioned in the original question. Follow these instructions carefully:

Here is the original question:
<original_question>
{question}
</original_question>

First, use a <scratchpad> to analyze the original question. In your scratchpad:
1. Identify all unique entities or aspects mentioned in the original question.
2. Determine if the specified number of alternatives ({num_alternatives}) is sufficient to cover all entities/aspects.
3. If not, calculate how many additional questions are needed.
4. Plan out consistent question structures for each entity/aspect.

After your scratchpad analysis, generate the alternative questions. The target number of questions to generate is {num_alternatives}. However, if you determine that more questions are necessary to cover all entities or aspects mentioned in the original question, generate additional questions to ensure comprehensive coverage.

When generating questions:
- Create concise variations, each focusing on a single aspect or entity from the original question.
- For questions mentioning multiple entities, create separate questions for each entity.
- Maintain consistency in question structure across different entities for fair comparison.
- Do not combine or compare entities in a single question.
- Generate questions that are short and optimized for semantic search, preferably less than 10 words each.
- Ensure each question is directly relevant and accurate to the original question's context.

Provide the alternative questions within <alternative_questions> tags, with each question on a new line. Do not number the questions or add any additional formatting.

Remember to prioritize quality and comprehensiveness over strictly adhering to the specified number of questions if necessary. Focus on creating questions that explore the key features or main aspects of each entity mentioned in the original question, rather than comparing them or discussing unrelated topics.
"""