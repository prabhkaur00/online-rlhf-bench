import streamlit as st

from config import (
    DATASET_DIR,
    GENERATED_TRACES_PATH,
    TRACE_COUNT,
)
from dataset import load_samples, split_question
from model_factory import get_model
from optimization_runner import run_update
from sampler import load_or_generate_traces

def load_data():
    samples = load_samples(DATASET_DIR)
    print("Loading model and traces...")
    model = get_model()
    traces = load_or_generate_traces(
        GENERATED_TRACES_PATH,
        samples,
        model,
    )
    return samples, traces


def main() -> None:
    st.title("MedQA RLHF Bench")
    samples, traces_map = load_data()
    if not samples:
        st.error("No samples loaded. Check DATASET_DIR.")
        return

    sample_ids = [s.sample_id for s in samples]
    selected = st.sidebar.selectbox("Sample", sample_ids)
    sample_lookup = {s.sample_id: s for s in samples}
    sample = sample_lookup.get(selected)
    if sample is None:
        st.warning("Selected sample not found; falling back to the first sample.")
        sample = samples[0]

    stem, options = split_question(sample.question)
    st.subheader("Flow")
    st.markdown("**Context**")
    st.write(stem)
    st.markdown("**Question**")
    if options:
        st.write("\n".join(options))
    else:
        st.write(sample.question)

    st.markdown("**Verified Traces**")
    traces = traces_map.get(selected)
    if traces is None:
        st.warning("Traces missing for selected sample; falling back to the first sample.")
        sample_id = samples[0].sample_id
        traces = traces_map.get(sample_id, [])
    for i, trace in enumerate(traces, start=1):
        st.text_area(f"Trace {i}", trace, height=140)

    st.subheader("Interaction")
    trace_labels = [f"Trace {i}" for i in range(1, TRACE_COUNT + 1)]
    def normalize_choice(choice, labels, fallback_index):
        if choice in labels:
            return choice
        if isinstance(choice, int) and 0 <= choice < len(labels):
            return labels[choice]
        return labels[min(fallback_index, len(labels) - 1)]

    if "best_choice" in st.session_state:
        st.session_state["best_choice"] = normalize_choice(
            st.session_state["best_choice"], trace_labels, 0
        )
    if "middle_choice" in st.session_state:
        st.session_state["middle_choice"] = normalize_choice(
            st.session_state["middle_choice"], trace_labels, 1
        )
    if "worst_choice" in st.session_state:
        st.session_state["worst_choice"] = normalize_choice(
            st.session_state["worst_choice"], trace_labels, 2
        )

    best = st.selectbox("Best", trace_labels, index=0, key="best_choice")
    middle = st.selectbox("Middle", trace_labels, index=1, key="middle_choice")
    worst = st.selectbox("Worst", trace_labels, index=2, key="worst_choice")

    best = normalize_choice(best, trace_labels, 0)
    middle = normalize_choice(middle, trace_labels, 1)
    worst = normalize_choice(worst, trace_labels, 2)

    ranking = [
        trace_labels.index(best),
        trace_labels.index(middle),
        trace_labels.index(worst),
    ]

    if len(set(ranking)) != 3:
        st.warning("Please choose distinct traces for Best/Middle/Worst.")
        return

    if st.button("Update Model"):
        print("Ranking received: best=%s middle=%s worst=%s" % (best, middle, worst))
        model = get_model()
        metrics = run_update(model, traces, ranking)
        st.subheader("Optimization Metrics")
        for key, value in metrics.items():
            st.write(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
