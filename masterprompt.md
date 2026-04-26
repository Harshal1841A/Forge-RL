# FORGE UI Design Master Prompt

When designing or updating UI components for FORGE, follow these core principles:

1. **Hierarchy and Prominence**: Critical information (like the CLAIM section) must be large, prominent, and positioned above auxiliary dashboard information. It is the main part of the investigation context.
2. **Readability over Aesthetics**: Glassmorphism and translucency are great for the modern spatial look (`backdrop-blur`, borders, glows), but backgrounds must be opaque enough (e.g., `bg-slate-900/60` or `/40` minimum, rather than `/15`) to ensure all text and data remains highly readable against the dynamic animated backgrounds.
3. **Motion reflects State**: Do not treat the interface as decorative UI. Do not over-animate. All motion must reflect backend state (IDLE, ACTIVE, EVENT, DECISION) using Framer Motion with spring physics.
4. **Cinematic but Controlled**: The UI must feel like a live distributed system—agents actively thinking with subtle, intelligent motion. It should not be flashy or distracting.
