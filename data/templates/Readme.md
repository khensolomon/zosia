# Template engine

## A Practical Rule of Thumb

Here is a general guide for how many different contexts (parallel sentences) you should aim for when introducing a new concept:

### For Nouns (especially proper nouns like names and places):

Minimum: 5-10 examples. Because our model has a copy mechanism, it's very good at learning to carry over words that don't need translation. A handful of examples is often enough for it to learn "when I see this pattern of subwords, I should copy it."

Ideal: 20+ examples. This will help the model understand how the noun interacts with different verbs and adjectives.

### For Adjectives and Adverbs:

Minimum: 10-15 examples. These words need a bit more context. The model needs to see an adjective used with several different nouns to understand what it modifies.

Ideal: 30-50+ examples. This will help it learn the nuances and avoid just memorizing a single phrase.

### For Verbs (The Most Important and Difficult):

Minimum: 20-30 examples. Verbs are the heart of a sentence. They change form (conjugation) and have complex relationships with subjects and objects. The model needs to see a verb in many different situations to learn its meaning.

Ideal: 50-100+ examples. To make the model truly robust, you want to show it the verb with different subjects (I, you, he, the man), different objects, and in different tenses if possible.

```cmd
I want water
I drink water
Water is good
I don’t like water
Where is the water?
This water is cold
Can I have water?
Water is necessary
I bought water
He spilled the water
```

tea

```tsv
pasalpa a lungdam hi	the man is happy
numeinu a lungdam hi	the woman is happy
suangtum a hoih hi	the rock is good
kei ka lungdam hi	I am happy
nang na lungdam hi	you are happy
amahpa a lungdam hi	he is happy
amahnu a lungdam hi	she is happy
pasalno a lungdam hi	the boy is happy
numeino a lungdam hi	the girl is happy
