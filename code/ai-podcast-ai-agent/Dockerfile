# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.11

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user app.py app.py
COPY --chown=user blog_summarizer.py blog_summarizer.py

# Expose the port that Gradio will run on
EXPOSE 7860

# Run the Gradio app
CMD ["python", "app.py"]
