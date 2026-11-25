# The agent's sandbox: a bare Python image with one command, `tsq`. No project code, no
# credentials, no database address: only GATEWAY_URL. Build context: this directory.
FROM python:3.12-alpine
COPY tsq /usr/local/bin/tsq
RUN chmod 755 /usr/local/bin/tsq
USER 1000
WORKDIR /tmp
CMD ["tail", "-f", "/dev/null"]
