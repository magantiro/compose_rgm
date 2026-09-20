FROM debian:bookworm-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        openbabel \
        python3 \
    && rm -rf /var/lib/apt/lists/*

ENV OMP_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/src:/workspace \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

ENTRYPOINT ["python3", "-m", "compose_v4.experiments.t4_nodistill_topology_four_call_local_runtime"]
