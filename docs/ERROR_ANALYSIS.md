# Observed generations and failure analysis

## Successful examples

| Input | Model output |
|---|---|
| Your idea is stupid. | Your idea is bad. |
| This explanation is garbage. | This explanation is bad. |
| This damn computer keeps crashing. | This computer keeps crashing. |
| What the hell are you doing? | What are you doing? |

## Failure cases (not manually corrected)

| Input | Observed output | Failure |
|---|---|---|
| Stop acting like an idiot. | Stop acting like an. | Truncated grammar |
| What a dumb decision! | What an decision! | Incorrect determiner |
| These idiots ruined the meeting. | These ruin ruined the meeting. | Redundancy and semantic error |
| She worked her ass off to prepare the presentation. | She worked her off to prepare the presentation. | Broken idiom |
| The stupid report says 18.5% instead of 15.8%. | The stupid report says 18.5% instead of 15. | Residual toxicity and broken number |

**Interpretation:** deleting offensive tokens is not enough when an entire phrase needs grammatical reworking; idioms, numbers and negation require more than subword overlap. The 60 initial + 40 additional diagnostic prompts are *nonrepresentative qualitative tests*. They cannot establish a population-level success rate.

**Suggested future work:** human ratings of toxicity, semantic fidelity and grammar; multiple training seeds; improved numerical/negation consistency; phrase-level rewriting and evaluation on held-out challenging expressions.
