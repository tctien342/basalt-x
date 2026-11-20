# API formats

These modules serialize the OpenAI Chat and Responses and Anthropic Messages formats and validate grammar, logit bias and structured output options. The HTTP handler selects the service and the canonical request model. Chat and Messages read that name from the service's request context while streaming.
