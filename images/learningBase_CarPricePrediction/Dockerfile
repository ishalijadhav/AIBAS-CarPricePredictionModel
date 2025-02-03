FROM busybox
RUN mkdir -p /learningBase/train /learningBase/validation
COPY data/training_data.csv /learningBase/train/
COPY data/test_data.csv /learningBase/validation/
COPY Readme.md /learningBase/Readme.md
# CMD cp -r data/training_data.csv /tmp/learningBase/train/ && \
#     cp -r data/test_data.csv /tmp/learningBase/validation/ && \
#     tail -f /dev/null