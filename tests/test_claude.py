## Test Claude API call

import anthropic

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY (or your `ant` login) automatically

response = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    messages=[{"role": "user", "content": "Explain what N3 sleep is in two sentences."}],
)

# The reply is a list of blocks; text blocks hold the answer.
for block in response.content:
    if block.type == "text":
        print(block.text)
