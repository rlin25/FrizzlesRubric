# Subplan 6: Length Limit

## Description
Reject prompts exceeding a fixed character or token limit.

## Steps
- Check if prompt exceeds 512 characters (or other fixed limit)
- If exceeded, return error or rejection response

## Input
- Raw prompt (string)

## Output
- Error/rejection response if limit exceeded
- Otherwise, continue processing 