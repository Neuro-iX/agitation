FROM python:3

# Set environment variables
ENV PATH="/root/.local/bin:${PATH}"

# Install agitation package and perform checks
RUN pip install --root-user-action=ignore agitation && \
    agitation manage check || true && \
    agitation --help || echo "Agitation not found"

# Labels
LABEL Author="Charles Bricout"
LABEL Version="v0.0.2"

# Help
CMD ["agitation", "--help"]