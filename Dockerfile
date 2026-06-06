FROM vllm/vllm-openai:latest

# Install audio dependencies for multimodal input
RUN pip install soundfile av

COPY apply_patch.py /tmp/apply_patch.py
RUN python3 /tmp/apply_patch.py
