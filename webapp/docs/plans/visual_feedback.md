# Visual Feedback (Lights)

- Display a row or grid of colored lights, one for each expert
- Each light changes color based on the expert's result
  - Green for 1, red for 0, yellow for 'e'
- Lights update immediately after receiving orchestrator API response
- Use React state to manage light colors
- Animate color transitions for visual effect
- Optionally, show expert names below each light 