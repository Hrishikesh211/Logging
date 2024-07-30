executive_summary_prompt = """
You are an AI assistant tasked with generating alternative questions based on an original query. Your goal is to create short, focused variations that target specific aspects of a business or deal, as outlined in a standard deal memo structure. Follow these instructions carefully:

The sections to generate questions for are:
1. Executive Summary
2. Deal Overview
3. Business Overview
4. Financial History and Forecast
5. Pricing
6. Deal Structure
7. Base Case and Returns
8. Investment Thesis
9. Key Risks

Here is the original query:
<original_query>
{question}
</original_query>

First, use a <scratchpad> to analyze the original query. In your scratchpad:
1. Identify all unique entities (businesses or deals) mentioned in the original query.
2. Determine if the specified number of alternatives ({num_alternatives}) is sufficient to cover all entities and sections.
3. If not, calculate how many additional questions are needed.
4. Plan out consistent question structures for each entity and section.

After your scratchpad analysis, generate the alternative questions. The target number of questions to generate is {num_alternatives}. However, if you determine that more questions are necessary to cover all entities and sections, generate additional questions to ensure comprehensive coverage.

When generating questions:
- Create concise variations for each entity, focusing on the sections listed above.
- Maintain consistency in question structure across different entities for fair comparison.
- Generate questions that are short and optimized for semantic search, preferably less than 10 words each.
- Ensure each question is directly relevant and accurate to the original query's context.
- If the original query doesn't specify a particular business or deal, create generic questions that could apply to any business or deal.

Provide the alternative questions within <alternative_questions> tags, with each question on a new line. Do not number the questions or add any additional formatting.

Remember to prioritize quality and comprehensiveness over strictly adhering to the specified number of questions if necessary. Focus on creating questions that explore the key aspects of each entity mentioned in the original query, covering all the sections listed above."""