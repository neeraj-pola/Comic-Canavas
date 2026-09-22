"""Per-user preference learning: the app learns which images a person
prefers from their A/B taps, shows it learning, and steers future
candidates toward what they like.

Modules: `features` (what we measure about an image), `model` (Bayesian
Bradley-Terry over those features), `axes` (candidates that differ on
purpose, so a tap teaches a direction), `state` (the model as the pipeline
uses it), `retrain` (rebuild the model + chart snapshots from the taps).
"""
