---
name: Web Research Prompt
description: Research the latest information on a topic using current web sources.
agent: "Web Research Agent"
argument-hint: Describe the topic, desired freshness, and any source preferences.
tools: [web, read, search]
---
You are researching current information on a topic.

Task:
<describe what needs to be researched>

Context:
- Topic: <topic>
- Freshness needed: <today, this week, this month, latest docs, etc.>
- Preferred sources: <official docs, news, blogs, market data, etc.>
- Constraints: <regions, dates, sources to avoid>

What I want:
1. Find the latest trustworthy information.
2. Prefer authoritative or primary sources.
3. Compare multiple sources if the topic is ambiguous.
4. Summarize clearly with links and dates.
5. Call out anything uncertain or outdated.

Output format:
- Findings
- Sources
- Confidence and caveats
- Next steps
