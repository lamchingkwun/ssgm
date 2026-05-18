# SSGM Core Engine

This directory contains the implementation of the Stability- and Safety-Governed Memory (SSGM) layer.

## Role in this repository

This is the **core implementation** of the governed-memory layer. It provides the components used by the experiment drivers and public evaluation pipeline.

## What lives here

The package includes components for:
- governed write admission
- access-scoped retrieval and filtering
- provenance-aware controls
- ledger-backed evidence tracking
- reconciliation and repair over mutable memory state

Read the root `README.md` for the package overview and the included data tooling.
