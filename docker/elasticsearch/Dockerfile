# Stage 1: build the Vietnamese analysis plugin zip from source.
# The plugin is not published to a Maven/plugin repository — the project's
# own install docs (references.md §4.3) point at a locally built
# target/releases/*.zip, so we build it here instead of trusting a mirror.
FROM maven:3.9-eclipse-temurin-21 AS plugin-builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the v9.3.0 tag, matching the plugin/Elasticsearch compatibility
# table in references.md §4.3 (plugin 9.3.0 <-> ES 9.0.0-9.3.0).
RUN curl -fsSL -o source.tar.gz \
    https://github.com/duydo/elasticsearch-analysis-vietnamese/archive/refs/tags/v9.3.0.tar.gz \
    && tar xzf source.tar.gz --strip-components=1 \
    && rm source.tar.gz

RUN mvn -q -DskipTests package

# Stage 2: official Elasticsearch image with the plugin installed.
FROM docker.elastic.co/elasticsearch/elasticsearch:9.3.0

COPY --from=plugin-builder \
    /build/target/releases/elasticsearch-analysis-vietnamese-9.3.0.zip \
    /tmp/elasticsearch-analysis-vietnamese-9.3.0.zip

RUN bin/elasticsearch-plugin install --batch \
    file:///tmp/elasticsearch-analysis-vietnamese-9.3.0.zip
