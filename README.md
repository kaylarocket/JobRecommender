# job_recommender_app

A new Flutter project.

## Evaluation Protocol

Model evaluation uses implicit feedback with leave-one-out splitting, candidate filtering by preferred location/target role, and negative sampling (N=99 by default, configurable via `--negative-sample-size`). Metrics are reported at K={1,5,10} in addition to the legacy @10 columns.

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.
